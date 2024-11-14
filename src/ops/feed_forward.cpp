#include "src/ops/feed_forward.h"

#include "src/core/command.h"
#include "src/core/common.h"
#include "src/core/gpu_device.h"
#include "src/core/pipeline.h"
#include "src/core/tensor.h"
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
  gate_op_.reset (new MatMul (dev_, command_, w1_, 1.0, .0, 1, 0,
                              transposed_weight_, FP16, dtype_));
  down_op_.reset (new MatMul (dev_, command_, w2_, 1.0, .0, 0, 0,
                              transposed_weight_, FP16, dtype_));
  up_op_.reset (new MatMul (dev_, command_, w3_, 1.0, .0, 0, 0,
                            transposed_weight_, FP16, dtype_));

  elemwise_op_.reset (new ElementWise (dev_, command_, 2, FP16));
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
  if (!(ret = up_op_->init ()).ok () || !(ret = down_op_->init ()).ok ()
      || !(ret = gate_op_->init ()).ok ()
      || !(ret = elemwise_op_->init ()).ok ())
    {
      return ret;
    }

  return absl::OkStatus ();
}

uint64_t
FeedForward::time () noexcept
{
  auto up_time = up_op_->time ();
  auto gate_time = gate_op_->time ();
  auto down_time = down_op_->time ();
  auto elem_time = elemwise_op_->time ();

#if __VKLLAMA_LOG_COST
  fprintf (stderr,
           "FeedForward: up time cost: %llu, gate time cost: %llu, down time "
           "cost: %llu, elemwise time cost: %llu\n",
           up_time, gate_time, down_time, elem_time);
#endif

  return std::max (up_time, gate_time) + down_time + elem_time;
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

  if (!(ret = up_op_->operator() (X)).ok ())
    {
      return ret;
    }

  t0_ = *ret;

  if (!(ret = gate_op_->operator() (X)).ok ())
    {
      return ret;
    }

  t1_ = *ret;

  t0_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  t0_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  t1_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  t1_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  t2_ = Tensor::like (t0_);
  VKLLAMA_STATUS_OK (t2_.create ());

  ret = elemwise_op_->operator() (t0_, t1_);
  if (!ret.ok ())
    {
      return ret;
    }

  t2_ = *ret;

  if (!(ret = down_op_->operator() (t2_)).ok ())
    {
      return ret;
    }

  ret->set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  ret->set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  return ret;
}
}

