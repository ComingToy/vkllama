#include "src/ops/feed_forward.h"

#include "src/core/command.h"
#include "src/core/common.h"
#include "src/core/gpu_device.h"
#include "src/core/pipeline.h"
#include "src/core/tensor.h"
#include "src/shaders/matmul_conf.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <cstdio>

namespace vkllama
{
FeedForward::FeedForward (GPUDevice *dev, Command *command, Tensor w1,
                          Tensor w2, Tensor w3, const bool transposed_weight,
                          const Tensor::DType dtype)
    : Op (dev, command), w1_ (w1), w2_ (w2), w3_ (w3), dtype_ (dtype),
      transposed_weight_ (transposed_weight)
{
  down_op_.reset (new MatMul (dev_, command_, w2_, 1.0, .0, 0, 0,
                              transposed_weight_, FP16, dtype_));
}

absl::Status
FeedForward::init () noexcept
{
  if (dtype_ != FP16 && dtype_ != Q8_0)
    {
      return absl::InvalidArgumentError (
          "FeedForward op: only fp16 and q8_0 are supported.");
    }

  if (w1_.dtype () != w2_.dtype () || w2_.dtype () != w3_.dtype ()
      || dtype_ != w1_.dtype ())
    {
      return absl::InvalidArgumentError (
          absl::StrFormat ("feed_forward op defined with %d dtype but "
                           "w1.dtype = %d, w2.dtype = %d, w3.dtype = %d",
                           int (dtype_), int (w1_.dtype ()),
                           int (w2_.dtype ()), int (w3_.dtype ())));
    }

  absl::Status ret;
  if (!(ret = down_op_->init ()).ok ())
    {
      return ret;
    }

  const auto *code = __get_ffn_up_and_gate_q8_0_comp_spv_code ();
  const auto size = __get_ffn_up_and_gate_q8_0_comp_spv_size ();

  Pipeline::ShaderInfo info
      = { 0, 4, sizeof (ShapeConstant) * 4, (uint32_t)dev_->subgroup_size (),
          1, 1 };

  up_gate_pipeline_.reset (new Pipeline (dev_, code, size, {}, info));

  if (!(ret = up_gate_pipeline_->init ()).ok ())
    {
      return ret;
    }

  return absl::OkStatus ();
}

uint64_t
FeedForward::time () noexcept
{
  auto down_time = down_op_->time ();
  auto elem_time = up_gate_pipeline_->time ();

#if __VKLLAMA_LOG_COST
  fprintf (stderr,
           "FeedForward: up time cost: %llu, gate time cost: %llu, down time "
           "cost: %llu, elemwise time cost: %llu\n",
           up_time, gate_time, down_time, elem_time);
#endif

  return down_time + elem_time;
}

absl::StatusOr<Tensor>
FeedForward::operator() (Tensor X) noexcept
{
  if (X.dtype () != FP16)
    {
      return absl::InvalidArgumentError (absl::StrFormat (
          "feed_forward op defined with FP16 input but input X.dtype() = %d",
          int (X.dtype ())));
    }

  absl::StatusOr<Tensor> ret;
  t0_ = Tensor (X.channels (), X.height (), w1_.height (), dev_, FP16, false);

  VKLLAMA_STATUS_OK (t0_.create ());

  uint32_t groupx = (uint32_t)t0_.width (), groupy = (uint32_t)t0_.height (),
           groupz = (uint32_t)t0_.channels ();

  groupx = (groupx + Q8_0_TILE_X_SIZE - 1) / Q8_0_TILE_X_SIZE;

  VKLLAMA_STATUS_OK (up_gate_pipeline_->set_group (groupx, groupy, groupz));

  auto constants = X.shape_constant () + w3_.shape_constant ()
                   + w1_.shape_constant () + t0_.shape_constant ();

  VKLLAMA_STATUS_OK (command_->record_pipeline (
      *up_gate_pipeline_, { X, w3_, w1_, t0_ }, constants));

  t0_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  t0_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  if (!(ret = (*down_op_) (t0_)).ok ())
    {
      return ret;
    }

  ret->set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  ret->set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return ret;
}
}

