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

absl::Status
Rope::init () noexcept
{
  if (dtype_ != FP16)
    {
      return absl::InvalidArgumentError (
          "Rope op: only fp16 dtype is supported.");
    }

  Pipeline::ShaderInfo shader_info_k
      = { 0, 2, 4 * sizeof (uint32_t), 16, 16, 1 };
  Pipeline::ShaderInfo shader_info_q
      = { 0, 2, 4 * sizeof (uint32_t), 16, 16, 1 };

  const auto *spv_code = __get_rope_fp16_comp_spv_code ();
  const auto spv_size = __get_rope_fp16_comp_spv_size ();

  pipeline_k_.reset (
      new Pipeline (dev_, spv_code, spv_size, {}, shader_info_k));

  pipeline_q_.reset (
      new Pipeline (dev_, spv_code, spv_size, {}, shader_info_q));

  absl::Status ret;
  if (!(ret = pipeline_k_->init ()).ok ()
      || !(ret = pipeline_q_->init ()).ok ())
    {
      return ret;
    }

  return absl::OkStatus ();
}

uint64_t
Rope::time () noexcept
{
  return std::max (pipeline_k_->time (), pipeline_q_->time ());
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
           groupy = (query.height () + 15) / 16, groupz = query.channels ();

  ret = pipeline_q_->set_group (groupx, groupy, groupz);
  if (!ret.ok ())
    {
      return ret;
    }

  ShaderConstants shape
      = { (uint32_t)query.channels (), (uint32_t)query.height (),
          (uint32_t)query.width (), (uint32_t)offset };
  ret = command_->record_pipeline (*pipeline_q_, { query, out_query }, shape);
  if (!ret.ok ())
    {
      return ret;
    }

  out_query.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out_query.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return out_query;
}

}

