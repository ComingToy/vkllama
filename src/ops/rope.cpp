#include "src/ops/rope.h"
#include "src/core/command.h"
#include "src/core/pipeline.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>
#include <vector>

namespace vkllama
{
Rope::Rope (GPUDevice *dev, Command *command, const int maxlen, const int dim,
            const Tensor::DType dtype)
    : Op (dev, command), maxlen_ (maxlen), dim_ (dim), dtype_ (dtype)
{
}

static void
compute_freq (size_t dim, size_t maxlen, std::vector<float> &freqc,
              std::vector<float> &freqs)
{
  std::vector<float> freq;
  std::generate_n (std::back_inserter (freq), dim / 2,
                   [dim, n = .0f] () mutable {
                     float f = 1.0f / std::pow (10000.0f, n / dim);
                     n += 2.0;
                     return f;
                   });

  // [seqlen, headim]
  freqs.resize (maxlen * dim / 2);
  freqc.resize (maxlen * dim / 2);

  for (int i = 0; i < maxlen; ++i)
    {
      for (int k = 0; k < dim / 2; ++k)
        {
          auto f = freq[k] * static_cast<float> (i);
          auto c = std::cos (f);
          auto s = std::sin (f);
          auto pos = i * dim / 2 + k;
          freqc[pos] = c;
          freqs[pos] = s;
        }
    }
}

absl::Status
Rope::init () noexcept
{
  if (dtype_ != FP16)
    {
      return absl::InvalidArgumentError (
          "Rope op: only fp16 dtype is supported.");
    }

  Pipeline::ShaderInfo shader_info_q
      = { 0, 4, 4 * sizeof (uint32_t), 16, 2, 1 };

  const auto *spv_code = __get_rope_fp16_comp_spv_code ();
  const auto spv_size = __get_rope_fp16_comp_spv_size ();

  pipeline_q_.reset (
      new Pipeline (dev_, spv_code, spv_size, {}, shader_info_q));

  absl::Status ret;
  if (!(ret = pipeline_q_->init ()).ok ())
    {
      return ret;
    }

  freqc_ = Tensor (1, 2 * maxlen_, dim_, dev_, FP32, false);
  freqs_ = Tensor (1, 2 * maxlen_, dim_, dev_, FP32, false);

  if (!(ret = freqc_.create ()).ok () || !(ret = freqs_.create ()).ok ())
    {
      return ret;
    }

  std::vector<float> freqc, freqs;
  compute_freq (dim_, 2 * maxlen_, freqc, freqs);

  ret = command_->upload ((const uint8_t *)freqc.data (),
                          freqc.size () * sizeof (__vkllama_fp16_t), freqc_);
  if (!ret.ok ())
    {
      return ret;
    }

  ret = command_->upload ((const uint8_t *)freqs.data (),
                          freqs.size () * sizeof (__vkllama_fp16_t), freqs_);
  if (!ret.ok ())
    {
      return ret;
    }

  ret = pipeline_q_->update_bindings ({ freqc_, freqs_ }, { 2, 3 });
  if (!ret.ok ())
    {
      return ret;
    }
  return absl::OkStatus ();
}

uint64_t
Rope::time () noexcept
{
  return pipeline_q_->time ();
}

absl::StatusOr<Tensor>
Rope::operator() (Tensor query, const size_t offset) noexcept
{
  if (query.width () != dim_ || query.height () > maxlen_)
    {
      return absl::InvalidArgumentError (absl::StrFormat (
          "rope shape error. input0.shape = (%zu, %zu, %zu)"
          " dim = %d, maxlen = %d",
          query.channels (), query.height (), query.width (), dim_, maxlen_));
    }

  if (query.dtype () != dtype_)
    {
      return absl::InvalidArgumentError (
          absl::StrFormat ("rope defined with dtype %d. but query.dtype = %d",
                           int (dtype_), int (query.dtype ())));
    }

  auto out_query = Tensor::like (query);
  auto ret = out_query.create ();
  if (!ret.ok ())
    {
      return ret;
    }

  uint32_t groupx = (query.width () / 2 + 15) / 16,
           groupy = (query.height () + 1) / 2, groupz = query.channels ();

  ret = pipeline_q_->set_group (groupx, groupy, groupz);
  if (!ret.ok ())
    {
      return ret;
    }

  ShaderConstants shape
      = { (uint32_t)query.channels (), (uint32_t)query.height (),
          (uint32_t)query.width (), (uint32_t)offset };
  ret = command_->record_pipeline (*pipeline_q_, { query, out_query },
                                   { 0, 1 }, shape);
  if (!ret.ok ())
    {
      return ret;
    }

  out_query.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out_query.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return out_query;
}

}

