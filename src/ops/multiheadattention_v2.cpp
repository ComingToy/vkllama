#include "src/ops/multiheadattention_v2.h"
#include "src/core/command.h"
#include "src/core/common.h"
#include "src/core/gpu_device.h"
#include "src/core/quants.h"
#include "src/ops/elementwise.h"
#include "src/ops/mat_mul.h"
#include "src/ops/rope.h"
#include "src/shaders/matmul_conf.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkllama
{
MultiHeadAttentionV2::MultiHeadAttentionV2 (
    GPUDevice *dev, Command *command, Tensor wk, Tensor wq, Tensor wv,
    Tensor wo, const int maxlen, const int dim, const bool transposed_weight,
    Tensor::DType dtype, const bool use_kvcache, const bool clip_output)
    : Op (dev, command), wk_ (wk), wq_ (wq), wv_ (wv), wo_ (wo),
      maxlen_ (maxlen), dim_ (dim), transposed_weight_ (transposed_weight),
      dtype_ (dtype), use_kvcache_ (use_kvcache), clip_output_ (clip_output)
{
}

absl::Status
MultiHeadAttentionV2::init () noexcept
{
  if (dtype_ != FP16)
    {
      return absl::InvalidArgumentError (
          "MultiHeadAttentionV2: only fp16 dtype is supported.");
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

  {
    Pipeline::ShaderInfo info
        = { 0, 7, sizeof (ShapeConstant) * 3, (uint32_t)dev_->subgroup_size (),
            1, 1 };

    const auto *kqv_code = __get_kqv_fp16_x_q8_0_comp_spv_code ();
    const auto kqv_size = __get_kqv_fp16_x_q8_0_comp_spv_size ();
    kqv_pipeline_.reset (new Pipeline (dev_, kqv_code, kqv_size, {}, info));
  }

  absl::Status ret;
  matmul_o_
      = std::make_unique<MatMul> (dev_, command_, wo_, 1.0, 0, 0, 0,
                                  transposed_weight_, FP16, wo_.dtype ());

  float attn_score_scale = 1.0f / std::sqrt (static_cast<float> (dim_));
  matmul_qk_ = std::make_unique<MatMul> (dev_, command_, attn_score_scale, 0,
                                         0, 0, 1, FP16, FP16);

  rope_q_ = std::make_unique<Rope> (dev_, command_, maxlen_, dim_, dtype_);
  rope_k_ = std::make_unique<Rope> (dev_, command_, maxlen_, dim_, dtype_);
  matmul_weighted_
      = std::make_unique<MatMul> (dev_, command_, 1.0, 0, 0, 0, 0, FP16, FP16);

  matmul_attn_score_ = std::make_unique<MatMul> (dev_, command_, 1.0, .0, 0, 0,
                                                 true, FP16, FP16);

  softmax_ = std::make_unique<Softmax> (dev_, command_, true, 1.0, dtype_);
  transpose_k_ = std::make_unique<Transpose> (dev_, command_, 0, dtype_);
  transpose_q_ = std::make_unique<Transpose> (dev_, command_, 0, dtype_);
  transpose_v_ = std::make_unique<Transpose> (dev_, command_, 0, dtype_);
  transpose_heads_ = std::make_unique<Transpose> (dev_, command_, 0, dtype_);

  if (!(ret = kqv_pipeline_->init ()).ok ()
      || !(ret = matmul_o_->init ()).ok ()
      || !(ret = matmul_qk_->init ()).ok ()
      || !(ret = matmul_weighted_->init ()).ok ()
      || !(ret = rope_q_->init ()).ok () || !(ret = rope_k_->init ()).ok ()
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
          = Tensor (wk_.width () / dim_, maxlen_, dim_, dev_, dtype_, false);
      vcache_
          = Tensor (wv_.width () / dim_, maxlen_, dim_, dev_, dtype_, false);

      if (!(ret = kcache_.create ()).ok () || !(ret = vcache_.create ()).ok ())
        {
          return ret;
        }

      update_kcache_op_
          = std::make_unique<UpdateKVCache> (dev_, command_, dtype_);
      update_vcache_op_
          = std::make_unique<UpdateKVCache> (dev_, command_, dtype_);

      if (!(ret = update_kcache_op_->init ()).ok ()
          || !(ret = update_vcache_op_->init ()).ok ())
        {
          return ret;
        }
    }

  if (clip_output_)
    {
      clip_output_op_.reset (new Slice (dev_, command_, dtype_));
      if (!(ret = clip_output_op_->init ()).ok ())
        {
          return ret;
        }
    }

  return absl::OkStatus ();
}

static bool __enable_debug_log = false;

absl::StatusOr<Tensor>
MultiHeadAttentionV2::operator() (Tensor X, const size_t offset) noexcept
{
  absl::Status ret;

  auto print_fn = [this] (auto... args) {
    if (!__enable_debug_log)
      return absl::OkStatus ();
    return command_->print_tensor_mean (args...);
  };

  VKLLAMA_STATUS_OK (print_fn ("multiheadattention input mean: ", X));

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

  {
    size_t c = X.channels (), h = X.height (),
           w = transposed_weight_ ? wk_.height () : wk_.width ();
    if (!(k_.channels () == c && k_.height () == h && k_.width () == w))
      {
        k_ = Tensor (X.channels (), X.height (),
                     transposed_weight_ ? wk_.height () : wk_.width (), dev_,
                     X.dtype (), false);

        q_ = Tensor::like (k_);
        v_ = Tensor::like (k_);

        VKLLAMA_STATUS_OK (k_.create ());
        VKLLAMA_STATUS_OK (q_.create ());
        VKLLAMA_STATUS_OK (v_.create ());
      }
  }

  Tensor q = q_;
  {
    uint32_t groupx
        = (k_.width () + Q8_0_KQV_TILE_X_SIZE - 1) / Q8_0_KQV_TILE_X_SIZE,
        groupy = k_.height (), groupz = k_.channels ();

    VKLLAMA_STATUS_OK (kqv_pipeline_->set_group (groupx, groupy, groupz));

    auto constants
        = X.shape_constant () + wk_.shape_constant () + k_.shape_constant ();

    VKLLAMA_STATUS_OK (command_->record_pipeline (
        *kqv_pipeline_, { X, wk_, wq_, wv_, k_, q, v_ }, constants));
  }

  // [seqlen, heads, dim]
  if (auto ret = k_.reshape (k_.height (), k_.width () / dim_, dim_);
      !ret.ok ())
    {
      return ret;
    }

  if (auto ret = q.reshape (q.height (), q.width () / dim_, dim_); !ret.ok ())
    {
      return ret;
    }

  if (auto ret = v_.reshape (v_.height (), v_.width () / dim_, dim_);
      !ret.ok ())
    {
      return ret;
    }

  // clip to last token
  if (clip_output_)
    {
      std::array<uint32_t, 3> starts
          = { uint32_t (q.channels () - 1), uint32_t (0), uint32_t (0) };
      std::array<uint32_t, 3> sizes
          = { uint32_t (1), uint32_t (q.height ()), uint32_t (q.width ()) };

      auto ret = (*clip_output_op_) (q, starts, sizes);
      VKLLAMA_STATUS_OK (ret);
      q = *ret;
    }

  //[heads, seqlen, dim]
  auto transposed_k = (*transpose_k_) (k_);
  auto transposed_q = (*transpose_q_) (q);
  auto transposed_v = (*transpose_v_) (v_);

  VKLLAMA_STATUS_OK (transposed_k);
  VKLLAMA_STATUS_OK (transposed_q);
  VKLLAMA_STATUS_OK (transposed_v);

  VKLLAMA_STATUS_OK (
      print_fn ("multiheadattention transposed k mean: ", *transposed_k));
  VKLLAMA_STATUS_OK (
      print_fn ("multiheadattention transposed q mean: ", *transposed_q));
  VKLLAMA_STATUS_OK (
      print_fn ("multiheadattention transposed v mean: ", *transposed_v));

  auto roped_q = (*rope_q_) (*transposed_q, offset);

  auto roped_k = (*rope_k_) (*transposed_k, offset);

  VKLLAMA_STATUS_OK (roped_k);
  VKLLAMA_STATUS_OK (roped_q);

  VKLLAMA_STATUS_OK (print_fn ("multiheadattention roped q mean: ", *roped_q));
  VKLLAMA_STATUS_OK (print_fn ("multiheadattention roped k mean: ", *roped_k));

  if (use_kvcache_)
    {
      auto &update_kcache_op = *update_kcache_op_;
      auto &update_vcache_op = *update_vcache_op_;

      auto ret = update_kcache_op (kcache_, *roped_k, (uint32_t)offset);
      VKLLAMA_STATUS_OK (ret);

      ret = update_vcache_op (vcache_, *transposed_v, (uint32_t)offset);
      VKLLAMA_STATUS_OK (ret);

      uint32_t read_len = std::min (
          (uint32_t)(offset + transposed_k->height ()), (uint32_t)maxlen_);

      roped_k = kcache_.view (kcache_.channels (), read_len, kcache_.width ());

      VKLLAMA_STATUS_OK (roped_k);

      transposed_v
          = vcache_.view (vcache_.channels (), read_len, vcache_.width ());
      VKLLAMA_STATUS_OK (transposed_v);
    }

  VKLLAMA_STATUS_OK (
      print_fn ("multiheadattention cached k mean: ", *transposed_k));
  VKLLAMA_STATUS_OK (
      print_fn ("multiheadattention cached v mean: ", *transposed_v));

  // [heads, seqlen, seqlen]
  auto attn_scores = (*matmul_qk_) (*roped_q, *roped_k);
  VKLLAMA_STATUS_OK (attn_scores);
  VKLLAMA_STATUS_OK (
      print_fn ("multiheadattention attn_scores mean: ", *attn_scores));

  // TODO: could be funsed
  auto softmax_offset = attn_scores->width () - attn_scores->height ();
  auto softmax_attn_scores = (*softmax_) (*attn_scores, softmax_offset);

  VKLLAMA_STATUS_OK (softmax_attn_scores);
  VKLLAMA_STATUS_OK (print_fn ("multiheadattention softmax_attn_scores mean: ",
                               *softmax_attn_scores));

  // [heads, seqlen, dim]
  auto heads = (*matmul_weighted_) (*softmax_attn_scores, *transposed_v);
  VKLLAMA_STATUS_OK (heads);
  VKLLAMA_STATUS_OK (print_fn ("multiheadattention heads mean: ", *heads));

  //[seqlen, heads, dim]
  auto concated = (*transpose_heads_) (*heads);
  VKLLAMA_STATUS_OK (concated);
  VKLLAMA_STATUS_OK (
      print_fn ("multiheadattention concated heads mean: ", *concated));

  //[1, seqlen, heads*dim]
  ret = concated->reshape (1, concated->channels (),
                           concated->height () * concated->width ());
  VKLLAMA_STATUS_OK (ret);

  auto out = (*matmul_o_) (*concated);
  VKLLAMA_STATUS_OK (out);
  VKLLAMA_STATUS_OK (print_fn ("multiheadattention output mean: ", *out));

  // __enable_debug_log = false;
  return out;
}

uint64_t
MultiHeadAttentionV2::time () noexcept
{
  auto kqv_cost = kqv_pipeline_->time ();
  auto transposed_cost = std::max (
      { transpose_k_->time (), transpose_q_->time (), transpose_v_->time () });
  uint64_t kvcache_cost = 0;

  if (use_kvcache_)
    {
      auto update_cost
          = std::max (update_kcache_op_->time (), update_vcache_op_->time ());
      kvcache_cost = update_cost;
    }

  auto rope_cost = rope_k_->time ();
  auto attn_score_cost = matmul_qk_->time ();
  auto softmax_cost = softmax_->time ();
  auto weighted_cost = matmul_weighted_->time ();
  auto transpose_head_cost = transpose_heads_->time ();
  auto output_cost = matmul_o_->time ();

#if __VKLLAMA_LOG_COST
  fprintf (
      stderr,
      "attn time: kqv_cost = %llu, transposed_cost = %llu, kvcache_cost = "
      "%llu, rope_cost = %llu, attn_score_cost = %llu, softmax cost = %llu, "
      "weighted_cost = %llu, transpose head cost = %llu, output cost = %llu\n",
      kqv_cost, transposed_cost, kvcache_cost, rope_cost, attn_score_cost,
      softmax_cost, weighted_cost, transpose_head_cost, output_cost);
#endif

  return kqv_cost + transposed_cost + kvcache_cost + rope_cost
         + attn_score_cost + softmax_cost + weighted_cost + transpose_head_cost
         + output_cost;
}
}
