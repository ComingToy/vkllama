#include "src/ops/multiheadattention_v2.h"
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

namespace vkllama
{
MultiHeadAttentionV2::MultiHeadAttentionV2 (
    GPUDevice *dev, Command *command, VkTensor wk, VkTensor wq, VkTensor wv,
    VkTensor wo, const int maxlen, const int dim, const bool transposed_weight,
    VkTensor::DType dtype, const bool use_kvcache)
    : Op (dev, command), wk_ (wk), wq_ (wq), wv_ (wv), wo_ (wo),
      maxlen_ (maxlen), dim_ (dim), transposed_weight_ (transposed_weight),
      dtype_ (dtype), use_kvcache_ (use_kvcache)
{
}

VkResult
MultiHeadAttentionV2::init () noexcept
{
  if (wk_.dtype () != dtype_ || wq_.dtype () != dtype_
      || wv_.dtype () != dtype_ || wo_.dtype () != dtype_)
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  if (wk_.channels () != wq_.channels () || wk_.height () != wq_.height ()
      || wk_.width () != wq_.width () || wq_.channels () != wv_.channels ()
      || wq_.width () != wv_.width () || wq_.height () != wv_.height ())
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  VkResult ret = VK_SUCCESS;
  matmul_k_ = std::make_unique<MatMul> (dev_, command_, wk_, 0, 0,
                                        transposed_weight_, dtype_);
  matmul_q_ = std::make_unique<MatMul> (dev_, command_, wq_, 0, 0,
                                        transposed_weight_, dtype_);
  matmul_v_ = std::make_unique<MatMul> (dev_, command_, wv_, 0, 0,
                                        transposed_weight_, dtype_);
  matmul_o_ = std::make_unique<MatMul> (dev_, command_, wo_, 0, 0,
                                        transposed_weight_, dtype_);

  matmul_qk_ = std::make_unique<MatMul> (dev_, command_, 0, 0, 1, dtype_);

  rope_ = std::make_unique<Rope> (dev_, command_, maxlen_, dim_, dtype_);
  matmul_weighted_
      = std::make_unique<MatMul> (dev_, command_, 0, 0, 0, dtype_);

  matmul_attn_score_
      = std::make_unique<MatMul> (dev_, command_, 0, 0, true, dtype_);
  scaled_ = std::make_unique<ElementWise> (dev_, command_, 2, dtype_);
  softmax_ = std::make_unique<Softmax> (dev_, command_, true, dtype_);
  transpose_k_ = std::make_unique<Transpose> (dev_, command_, 0, dtype_);
  transpose_q_ = std::make_unique<Transpose> (dev_, command_, 0, dtype_);
  transpose_v_ = std::make_unique<Transpose> (dev_, command_, 0, dtype_);
  transpose_heads_ = std::make_unique<Transpose> (dev_, command_, 0, dtype_);

  if ((ret = matmul_k_->init ()) != VK_SUCCESS
      || (ret = matmul_q_->init ()) != VK_SUCCESS
      || (ret = matmul_v_->init ()) != VK_SUCCESS
      || (ret != matmul_o_->init ()) != VK_SUCCESS
      || (ret = matmul_qk_->init ()) != VK_SUCCESS
      || (ret = matmul_weighted_->init ()) != VK_SUCCESS
      || (ret = rope_->init ()) != VK_SUCCESS
      || (ret = matmul_attn_score_->init ()) != VK_SUCCESS
      || (ret = transpose_k_->init ()) != VK_SUCCESS
      || (ret = transpose_q_->init ()) != VK_SUCCESS
      || (ret = transpose_v_->init ()) != VK_SUCCESS
      || (ret = transpose_heads_->init ()) != VK_SUCCESS
      || (ret = scaled_->init ()) != VK_SUCCESS
      || (ret = softmax_->init ()) != VK_SUCCESS)
    {
      return ret;
    }

  if (use_kvcache_)
    {
      kcache_ = VkTensor (wk_.channels (), maxlen_, dim_, dev_, dtype_, false);
      vcache_ = VkTensor (wv_.channels (), maxlen_, dim_, dev_, dtype_, false);

      if ((ret = kcache_.create ()) != VK_SUCCESS
          || (ret = vcache_.create ()) != VK_SUCCESS)
        {
          return ret;
        }

      update_kcache_op_
          = std::make_unique<UpdateKVCache> (dev_, command_, dtype_);
      update_vcache_op_
          = std::make_unique<UpdateKVCache> (dev_, command_, dtype_);

      kcache_slice_op_ = std::make_unique<Slice> (dev_, command_, dtype_);
      vcache_slice_op_ = std::make_unique<Slice> (dev_, command_, dtype_);
      if ((ret = update_kcache_op_->init ()) != VK_SUCCESS
          || (ret = update_vcache_op_->init ()) != VK_SUCCESS
          || (ret = kcache_slice_op_->init ()) != VK_SUCCESS
          || (ret = vcache_slice_op_->init ()) != VK_SUCCESS)
        {
          return ret;
        }
    }

  return VK_SUCCESS;
}

VkResult
MultiHeadAttentionV2::operator() (VkTensor X, VkTensor &out,
                                  const size_t offset) noexcept
{
  VkResult ret = VK_SUCCESS;

  if (X.dtype () != dtype_)
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  if ((!transposed_weight_ && wv_.height () != X.width ())
      || (transposed_weight_ && wv_.width () != X.width ()))
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  tmp_tensors_.clear ();

  VkTensor k, q, v;
  if ((ret = (*matmul_k_) (X, k)) != VK_SUCCESS
      || (ret = (*matmul_q_) (X, q)) != VK_SUCCESS
      || (ret = (*matmul_v_) (X, v)) != VK_SUCCESS)
    {
      return ret;
    }

  tmp_tensors_.push_back (k);
  tmp_tensors_.push_back (q);
  tmp_tensors_.push_back (v);

  // [seqlen, heads, dim]
  k.reshape (k.height (), k.width () / dim_, dim_);
  q.reshape (q.height (), q.width () / dim_, dim_);
  v.reshape (v.height (), v.width () / dim_, dim_);

  VkTensor transposed_k, transposed_q, transposed_v;
  if ((ret = (*transpose_k_) (k, transposed_k)) != VK_SUCCESS)
    {
      return ret;
    }

  if ((ret = (*transpose_q_) (q, transposed_q)) != VK_SUCCESS)
    {
      return ret;
    }

  if ((ret = (*transpose_v_) (v, transposed_v)) != VK_SUCCESS)
    {
      return ret;
    }

  tmp_tensors_.push_back (transposed_k);
  tmp_tensors_.push_back (transposed_q);
  tmp_tensors_.push_back (transposed_v);

  VkTensor roped_k, roped_q;
  if ((ret = (*rope_) (transposed_q, transposed_k, roped_q, roped_k, offset))
      != VK_SUCCESS)
    {
      return ret;
    }

  tmp_tensors_.push_back (roped_k);
  tmp_tensors_.push_back (roped_q);

  // [heads, seqlen, seqlen]
  VkTensor attn_scores;
  if ((ret = (*matmul_qk_) (roped_q, roped_k, attn_scores)) != VK_SUCCESS)
    {
      return ret;
    }

  tmp_tensors_.push_back (attn_scores);

  // TODO: could be funsed
  VkTensor scaled_attn_scores;
  float attn_score_scale = 1.0f / std::sqrt (static_cast<float> (dim_));
  if ((ret = (*scaled_) (attn_scores, attn_score_scale, scaled_attn_scores))
      != VK_SUCCESS)
    {
      return ret;
    }

  tmp_tensors_.push_back (scaled_attn_scores);

  VkTensor softmax_attn_scores;

  if ((ret = (*softmax_) (scaled_attn_scores, softmax_attn_scores, offset))
      != VK_SUCCESS)
    {
      return ret;
    }

  tmp_tensors_.push_back (softmax_attn_scores);

  // [heads, seqlen, dim]
  VkTensor heads;
  if ((ret = (*matmul_weighted_) (softmax_attn_scores, transposed_v, heads))
      != VK_SUCCESS)
    {
      return ret;
    }
  tmp_tensors_.push_back (heads);

  //[seqlen, heads, dim]
  VkTensor concated;
  if ((ret = (*transpose_heads_) (heads, concated)) != VK_SUCCESS)
    {
      return ret;
    }

  //[1, seqlen, heads*dim]
  concated.reshape (1, concated.channels (),
                    concated.height () * concated.width ());
  tmp_tensors_.push_back (concated);

  if ((ret = (*matmul_o_) (concated, out)) != VK_SUCCESS)
    {
      return ret;
    }

  return VK_SUCCESS;
}

uint64_t
MultiHeadAttentionV2::time () noexcept
{
  return 0;
}
}
