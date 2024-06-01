#include "src/ops/feed_forward.h"

#include "src/core/command.h"
#include "src/core/gpu_device.h"
#include "src/core/pipeline.h"
#include "src/core/tensor.h"
#include "src/shaders/vkllama_comp_shaders.h"

FeedForward::FeedForward (GPUDevice *dev, Command *command, VkTensor w1,
                          VkTensor w2, VkTensor w3,
                          const bool transposed_weight,
                          const VkTensor::DType dtype)
    : Op (dev, command), w1_ (w1), w2_ (w2), w3_ (w3), dtype_ (dtype),
      transposed_weight_ (transposed_weight)
{

  gate_op_.reset (
      new MatMul (dev_, command_, w1_, 1, 0, transposed_weight_, dtype_));
  down_op_.reset (
      new MatMul (dev_, command_, w2_, 0, 0, transposed_weight_, dtype_));
  up_op_.reset (
      new MatMul (dev_, command_, w3_, 0, 0, transposed_weight_, dtype_));

  elemwise_op_.reset (new ElementWise (dev_, command_, 2, dtype_));
}

VkResult
FeedForward::init () noexcept
{
  if (w1_.dtype () != w2_.dtype () || w2_.dtype () != w3_.dtype ()
      || dtype_ != w1_.dtype ())
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  VkResult ret = VK_SUCCESS;
  if ((ret = up_op_->init ()) != VK_SUCCESS
      || (ret = down_op_->init ()) != VK_SUCCESS
      || (ret = gate_op_->init ()) != VK_SUCCESS
      || (ret = elemwise_op_->init ()))
    {
      return ret;
    }

  return VK_SUCCESS;
}

uint64_t
FeedForward::time () noexcept
{
  return std::max (up_op_->time (), gate_op_->time ()) + down_op_->time ()
         + elemwise_op_->time ();
}

VkResult
FeedForward::operator() (VkTensor X, VkTensor &output) noexcept
{
  if (X.dtype () != dtype_)
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  VkResult ret = VK_SUCCESS;

  if ((ret = up_op_->operator() (X, t0_)) != VK_SUCCESS)
    {
      return ret;
    }

  if ((ret = gate_op_->operator() (X, t1_)) != VK_SUCCESS)
    {
      return ret;
    }

  t0_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  t0_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  t1_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  t1_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  t2_ = VkTensor::like (t0_);

  if ((ret = t2_.create ()) != VK_SUCCESS)
    {
      return ret;
    }

  ret = elemwise_op_->operator() (t0_, t1_, t2_);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  if ((ret = down_op_->operator() (t2_, output)) != VK_SUCCESS)
    {
      return ret;
    }

  output.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  output.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  return VK_SUCCESS;
}
