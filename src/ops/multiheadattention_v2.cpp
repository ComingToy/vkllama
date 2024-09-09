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

absl::Status
MultiHeadAttentionV2::init () noexcept
{
  if (wk_.dtype () != dtype_ || wq_.dtype () != dtype_
      || wv_.dtype () != dtype_ || wo_.dtype () != dtype_)
    {
      return absl::InvalidArgumentError (absl::StrFormat (
          "dtype of weights error. wk.dtype = %d, wv.dtype = %d, wq.dtype = "
          "%d, wo.dtype = %d",
          wk_.dtype (), wq_.dtype (), wv_.dtype (), wo_.dtype ()));
    }

  if (wk_.channels () != wq_.channels () || wk_.height () != wq_.height ()
      || wk_.width () != wq_.width () || wq_.channels () != wv_.channels ()
      || wq_.width () != wv_.width () || wq_.height () != wv_.height ())
    {
      return absl::InvalidArgumentError (absl::StrFormat (
          "shape of weights error"
          "wk.shape = (%zu, %zu, %zu), wq.shape = (%zu, %zu, %zu), wv.shape = "
          "(%zu, %zu, %zu), wo.shape = (%zu, %zu, %zu)",
          wk_.channels (), wk_.height (), wk_.width (), wq_.channels (),
          wq_.height (), wq_.width (), wv_.channels (), wv_.height (),
          wv_.width (), wo_.channels (), wo_.height (), wo_.width ()));
    }

  absl::Status ret;
  matmul_k_ = std::make_unique<MatMul> (dev_, command_, wk_, 1.0, .0, 0, 0,
                                        transposed_weight_, dtype_);
  matmul_q_ = std::make_unique<MatMul> (dev_, command_, wq_, 1.0, .0, 0, 0,
                                        transposed_weight_, dtype_);
  matmul_v_ = std::make_unique<MatMul> (dev_, command_, wv_, 1.0, .0, 0, 0,
                                        transposed_weight_, dtype_);
  matmul_o_ = std::make_unique<MatMul> (dev_, command_, wo_, 1.0, 0, 0, 0,
                                        transposed_weight_, dtype_);

  float attn_score_scale = 1.0f / std::sqrt (static_cast<float> (dim_));
  matmul_qk_ = std::make_unique<MatMul> (dev_, command_, attn_score_scale, 0,
                                         0, 0, 1, dtype_);

  rope_ = std::make_unique<Rope> (dev_, command_, maxlen_, dim_, dtype_);
  matmul_weighted_
      = std::make_unique<MatMul> (dev_, command_, 1.0, 0, 0, 0, 0, dtype_);

  matmul_attn_score_
      = std::make_unique<MatMul> (dev_, command_, 1.0, .0, 0, 0, true, dtype_);

  softmax_ = std::make_unique<Softmax> (dev_, command_, true, 1.0, dtype_);
  transpose_k_ = std::make_unique<Transpose> (dev_, command_, 0, dtype_);
  transpose_q_ = std::make_unique<Transpose> (dev_, command_, 0, dtype_);
  transpose_v_ = std::make_unique<Transpose> (dev_, command_, 0, dtype_);
  transpose_heads_ = std::make_unique<Transpose> (dev_, command_, 0, dtype_);

  if (!(ret = matmul_k_->init ()).ok () || !(ret = matmul_q_->init ()).ok ()
      || !(ret = matmul_v_->init ()).ok () || !(ret = matmul_o_->init ()).ok ()
      || !(ret = matmul_qk_->init ()).ok ()
      || !(ret = matmul_weighted_->init ()).ok ()
      || !(ret = rope_->init ()).ok ()
      || !(ret = matmul_attn_score_->init ()).ok ()
      || !(ret = transpose_k_->init ()).ok ()
      || !(ret = transpose_q_->init ()).ok ()
      || !(ret = transpose_v_->init ()).ok ()
      || !(ret = transpose_heads_->init ()).ok ()
      || !(ret = softmax_->init ()).ok ())
    {
      return ret;
    }

  if (use_kvcache_)
    {
      kcache_
          = VkTensor (wk_.width () / dim_, maxlen_, dim_, dev_, dtype_, false);
      vcache_
          = VkTensor (wv_.width () / dim_, maxlen_, dim_, dev_, dtype_, false);

      if (!(ret = kcache_.create ()).ok () || !(ret = vcache_.create ()).ok ())
        {
          return ret;
        }

      update_kcache_op_
          = std::make_unique<UpdateKVCache> (dev_, command_, dtype_);
      update_vcache_op_
          = std::make_unique<UpdateKVCache> (dev_, command_, dtype_);

      kcache_read_op_ = std::make_unique<ReadKVCache> (dev_, command_);
      vcache_read_op_ = std::make_unique<ReadKVCache> (dev_, command_);
      if (!(ret = update_kcache_op_->init ()).ok ()
          || !(ret = update_vcache_op_->init ()).ok ()
          || !(ret = kcache_read_op_->init ()).ok ()
          || !(ret = vcache_read_op_->init ()).ok ())
        {
          return ret;
        }
    }

  return absl::OkStatus ();
}

absl::Status
MultiHeadAttentionV2::operator() (VkTensor X, VkTensor &out,
                                  const size_t offset) noexcept
{
  absl::Status ret;

  if (X.dtype () != dtype_)
    {
      return absl::InvalidArgumentError (
          absl::StrFormat ("multiheadattention op defined with %d dtype but "
                           "input0's dtype = %d",
                           int (dtype_), int (X.dtype ())));
    }

  if ((!transposed_weight_ && wv_.height () != X.width ())
      || (transposed_weight_ && wv_.width () != X.width ()))
    {
      return absl::InvalidArgumentError (
          absl::StrFormat ("shape error. wv.shape = (%zu, %zu, %zu), "
                           "input0.shape = (%zu, %zu, %zu)",
                           wv_.channels (), wv_.height (), wv_.width (),
                           X.channels (), X.height (), X.width ()));
    }

  tmp_tensors_.clear ();

  VkTensor k, q, v;
  if (!(ret = (*matmul_k_) (X, k)).ok () || !(ret = (*matmul_q_) (X, q)).ok ()
      || !(ret = (*matmul_v_) (X, v)).ok ())
    {
      return ret;
    }

  tmp_tensors_.push_back (k);
  tmp_tensors_.push_back (q);
  tmp_tensors_.push_back (v);

  // [seqlen, heads, dim]
  if (auto ret = k.reshape (k.height (), k.width () / dim_, dim_); !ret.ok ())
    {
      return ret;
    }
  if (auto ret = q.reshape (q.height (), q.width () / dim_, dim_); !ret.ok ())
    {
      return ret;
    }

  if (auto ret = v.reshape (v.height (), v.width () / dim_, dim_); !ret.ok ())
    {
      return ret;
    }

  //[heads, seqlen, dim]
  VkTensor transposed_k, transposed_q, transposed_v;
  if (!(ret = (*transpose_k_) (k, transposed_k)).ok ())
    {
      return ret;
    }

  if (!(ret = (*transpose_q_) (q, transposed_q)).ok ())
    {
      return ret;
    }

  if (!(ret = (*transpose_v_) (v, transposed_v)).ok ())
    {
      return ret;
    }

  tmp_tensors_.push_back (transposed_k);
  tmp_tensors_.push_back (transposed_q);
  tmp_tensors_.push_back (transposed_v);

  if (use_kvcache_)
    {
      auto &update_kcache_op = *update_kcache_op_;
      auto &update_vcache_op = *update_vcache_op_;
      auto &kcache_read_op = *kcache_read_op_;
      auto &vcache_read_op = *vcache_read_op_;

      if (!(ret = update_kcache_op (kcache_, transposed_k, (uint32_t)offset))
               .ok ())
        {
          return ret;
        }

      if (!(ret = update_vcache_op (vcache_, transposed_v, (uint32_t)offset))
               .ok ())
        {
          return ret;
        }

      uint32_t read_offset = offset >= (size_t)maxlen_
                                 ? (uint32_t)offset % (uint32_t)maxlen_
                                 : 0;

      uint32_t read_len = std::min (
          (uint32_t)(offset + transposed_k.height ()), (uint32_t)maxlen_);

      ret = kcache_read_op (kcache_, read_offset, read_len, transposed_k);

      if (!ret.ok ())
        {
          return ret;
        }

      ret = vcache_read_op (vcache_, read_offset, read_len, transposed_v);

      if (!ret.ok ())
        {
          return ret;
        }

      tmp_tensors_.push_back (transposed_k);
      tmp_tensors_.push_back (transposed_v);
    }

  VkTensor roped_k, roped_q;
  if (!(ret = (*rope_) (transposed_q, transposed_k, roped_q, roped_k, offset))
           .ok ())
    {
      return ret;
    }

  tmp_tensors_.push_back (roped_k);
  tmp_tensors_.push_back (roped_q);

  // [heads, seqlen, seqlen]
  VkTensor attn_scores;
  if (!(ret = (*matmul_qk_) (roped_q, roped_k, attn_scores)).ok ())
    {
      return ret;
    }

  tmp_tensors_.push_back (attn_scores);

  // TODO: could be funsed
  VkTensor softmax_attn_scores;

  if (!(ret = (*softmax_) (attn_scores, softmax_attn_scores,
                           attn_scores.width () - attn_scores.height ()))
           .ok ())
    {
      return ret;
    }

  tmp_tensors_.push_back (softmax_attn_scores);

  // [heads, seqlen, dim]
  VkTensor heads;
  if (!(ret = (*matmul_weighted_) (softmax_attn_scores, transposed_v, heads))
           .ok ())
    {
      return ret;
    }
  tmp_tensors_.push_back (heads);

  //[seqlen, heads, dim]
  VkTensor concated;
  if (!(ret = (*transpose_heads_) (heads, concated)).ok ())
    {
      return ret;
    }

  //[1, seqlen, heads*dim]
  ret = concated.reshape (1, concated.channels (),
                          concated.height () * concated.width ());
  if (!ret.ok ())
    {
      return ret;
    }

  tmp_tensors_.push_back (concated);

  if (!(ret = (*matmul_o_) (concated, out)).ok ())
    {
      return ret;
    }

  return absl::OkStatus ();
}

uint64_t
MultiHeadAttentionV2::time () noexcept
{
  auto kqv_cost = std::max (
      { matmul_k_->time (), matmul_q_->time (), matmul_v_->time () });
  auto transposed_cost = std::max (
      { transpose_k_->time (), transpose_q_->time (), transpose_v_->time () });
  uint64_t kvcache_cost = 0;

  if (use_kvcache_)
    {
      auto update_cost
          = std::max (update_kcache_op_->time (), update_vcache_op_->time ());
      auto slice_cost
          = std::max (kcache_read_op_->time (), vcache_read_op_->time ());
      kvcache_cost = update_cost + slice_cost;
    }

  auto rope_cost = rope_->time ();
  auto attn_score_cost = matmul_qk_->time ();
  auto softmax_cost = softmax_->time ();
  auto weighted_cost = matmul_weighted_->time ();
  auto transpose_head_cost = transpose_heads_->time ();
  auto output_cost = matmul_o_->time ();

  // fprintf (
  //     stderr,
  //     "attn time: kqv_cost = %llu, transposed_cost = %llu, kvcache_cost = "
  //     "%llu, rope_cost = %llu, attn_score_cost = %llu, softmax cost = %llu,
  //     " "weighted_cost = %llu, transpose head cost = %llu, output cost =
  //     %llu\n", kqv_cost, transposed_cost, kvcache_cost, rope_cost,
  //     attn_score_cost, softmax_cost, weighted_cost, transpose_head_cost,
  //     output_cost);
  return kqv_cost + transposed_cost + kvcache_cost + rope_cost
         + attn_score_cost + softmax_cost + weighted_cost + transpose_head_cost
         + output_cost;
}
}
