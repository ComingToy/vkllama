#include "src/ops/multiheadattention.h"
#include "src/core/command.h"
#include "src/core/gpu_device.h"
#include "src/ops/elementwise.h"
#include "src/ops/mat_mul.h"
#include "src/ops/rope.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

MultiHeadAttention::MultiHeadAttention (
    GPUDevice *dev, Command *command, std::vector<VkTensor> const &Wk,
    std::vector<VkTensor> const &Wq, std::vector<VkTensor> const &Wv,
    const VkTensor Wo, const int maxlen, const int dim,
    const bool transposed_weight, const VkTensor::DType dtype)
    : Op (dev, command), wk_ (Wk), wq_ (Wq), wv_ (Wv), wo_ (Wo),
      maxlen_ (maxlen), dim_ (dim), transposed_weight_ (transposed_weight),
      dtype_ (dtype)
{
}

VkResult
MultiHeadAttention::init () noexcept
{
  if (wk_.size () != wq_.size () || wq_.size () != wv_.size ())
    {
      return VK_ERROR_UNKNOWN;
    }

  if (std::any_of (wk_.cbegin (), wk_.cend (),
                   [this] (auto &x) { return dtype_ != x.dtype (); })
      || std::any_of (wq_.cbegin (), wq_.cend (),
                      [this] (auto &x) { return x.dtype () != dtype_; })
      || std::any_of (wv_.cbegin (), wv_.cend (),
                      [this] (auto &x) { return x.dtype () != dtype_; })
      || wo_.dtype () != dtype_)
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  VkResult ret = VK_SUCCESS;
  out_matmul_ = std::make_unique<MatMul> (dev_, command_, wo_, 0, 0,
                                          transposed_weight_, dtype_);
  if ((ret = out_matmul_->init ()) != VK_SUCCESS)
    {
      return ret;
    }

  concat_
      = std::make_unique<Concat> (dev_, command_, wq_.size (), VkTensor::FP16);

  for (size_t i = 0; i < wq_.size (); ++i)
    {
      auto k_matmul = std::make_unique<MatMul> (dev_, command_, wk_[i], 0, 0,
                                                transposed_weight_, dtype_);
      auto q_matmul = std::make_unique<MatMul> (dev_, command_, wq_[i], 0, 0,
                                                transposed_weight_, dtype_);
      auto v_matmul = std::make_unique<MatMul> (dev_, command_, wv_[i], 0, 0,
                                                transposed_weight_, dtype_);
      auto weighted_matmul
          = std::make_unique<MatMul> (dev_, command_, 0, 0, 0, dtype_);

      auto rope
          = std::make_unique<Rope> (dev_, command_, maxlen_, dim_, dtype_);
      auto attn_score
          = std::make_unique<MatMul> (dev_, command_, 0, 0, 1, dtype_);
      auto elementwise
          = std::make_unique<ElementWise> (dev_, command_, 2, dtype_);
      auto softmax = std::make_unique<Softmax> (dev_, command_, true, dtype_);

      if ((ret = k_matmul->init ()) != VK_SUCCESS
          || (ret = q_matmul->init ()) != VK_SUCCESS
          || (ret = v_matmul->init ()) != VK_SUCCESS
          || (ret = weighted_matmul->init ()) != VK_SUCCESS
          || (ret = rope->init ()) != VK_SUCCESS
          || (ret = attn_score->init ()) != VK_SUCCESS
          || (ret = elementwise->init ()) != VK_SUCCESS
          || (ret = softmax->init ()) != VK_SUCCESS
          || (ret = out_matmul_->init ()) != VK_SUCCESS
          || (ret = concat_->init ()) != VK_SUCCESS)
        {
          return ret;
        }

      k_ops_.push_back (std::move (k_matmul));
      q_ops_.push_back (std::move (q_matmul));
      v_ops_.push_back (std::move (v_matmul));
      weighted_ops_.push_back (std::move (weighted_matmul));
      rope_ops_.push_back (std::move (rope));
      attn_ops_.push_back (std::move (attn_score));
      elementwise_ops_.push_back (std::move (elementwise));
      softmax_ops_.push_back (std::move (softmax));
    }

  return VK_SUCCESS;
}

VkResult
MultiHeadAttention::operator() (VkTensor X, VkTensor &output) noexcept
{
  VkResult ret = VK_SUCCESS;
  std::vector<VkTensor> head_tensors;
  std::vector<VkTensor> ks, qs, vs, sacled_attns, softmax_attns;
  VkTensor input = X;
  tmp_tensors_.clear ();

  if (X.dtype () != dtype_)
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  for (size_t i = 0; i < wq_.size (); ++i)
    {
      auto wk = wk_[i];
      auto wq = wq_[i];
      auto wv = wv_[i];
      if (wk.channels () != wq.channels () || wk.height () != wq.height ()
          || wk.width () != wq.width () || wq.channels () != wv.channels ()
          || wq.height () != wv.height () || wq.width () != wv.width ()
          || wv.channels () != input.channels ()
          || (!transposed_weight_ && wv.height () != input.width ())
          || (transposed_weight_ && wv.width () != input.width ()))
        {
          return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }

      VkTensor k, q, v;
      auto &matmul_k = *k_ops_[i];
      auto &matmul_q = *q_ops_[i];
      auto &matmul_v = *v_ops_[i];
      auto &weighted_matmul = *weighted_ops_[i];
      auto &rope = *rope_ops_[i];
      auto &attn_score = *attn_ops_[i];
      auto &scale_op = *elementwise_ops_[i];
      auto &softmax_op = *softmax_ops_[i];

      if ((ret = matmul_k (input, k)) != VK_SUCCESS
          || (ret = matmul_q (input, q)) != VK_SUCCESS
          || (ret = matmul_v (input, v)) != VK_SUCCESS)
        {
          return ret;
        }

      vs.push_back (v);
      tmp_tensors_.push_back (k);
      tmp_tensors_.push_back (q);
      tmp_tensors_.push_back (v);
      VkTensor roped_q, roped_k;
      if ((ret = rope (q, k, roped_q, roped_k)) != VK_SUCCESS)
        {
          return ret;
        }
      ks.push_back (roped_k);
      qs.push_back (roped_q);
      tmp_tensors_.push_back (roped_q);
      tmp_tensors_.push_back (roped_k);

      VkTensor attn_scores;
      if ((ret = attn_score (roped_q, roped_k, attn_scores)) != VK_SUCCESS)
        {
          return ret;
        }
      tmp_tensors_.push_back (attn_scores);

      VkTensor scaled_attn_scores;
      float attn_score_scale = 1.0f / std::sqrt (static_cast<float> (dim_));
      if ((ret = scale_op (attn_scores, attn_score_scale, scaled_attn_scores))
          != VK_SUCCESS)
        {
          return ret;
        }
      tmp_tensors_.push_back (scaled_attn_scores);
      sacled_attns.push_back (scaled_attn_scores);

      VkTensor softmax_attn_scores;
      if ((ret = softmax_op (scaled_attn_scores, softmax_attn_scores))
          != VK_SUCCESS)
        {
          return ret;
        }
      softmax_attns.push_back (softmax_attn_scores);
      tmp_tensors_.push_back (softmax_attn_scores);

      VkTensor head;
      if ((ret = weighted_matmul (softmax_attn_scores, v, head)) != VK_SUCCESS)
        {
          return ret;
        }
      tmp_tensors_.push_back (head);
      head_tensors.push_back (head);
    }

  VkTensor concated;
  if ((ret = concat_->operator() (head_tensors, concated)) != VK_SUCCESS)
    {
      return ret;
    }

  tmp_tensors_.push_back (concated);
#if 1
  if ((ret = out_matmul_->operator() (concated, output)) != VK_SUCCESS)
    {
      return ret;
    }
#endif

  return VK_SUCCESS;
}

uint64_t
MultiHeadAttention::time () noexcept
{
  return 0;
}
