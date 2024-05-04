#include "src/ops/feed_forward.h"

#include "src/core/command.h"
#include "src/core/gpu_device.h"
#include "src/core/pipeline.h"
#include "src/core/tensor.h"
#include "src/shaders/vkllama_comp_shaders.h"

FeedForward::FeedForward (GPUDevice *dev, Command *command, VkTensor w1,
                          VkTensor w2, VkTensor w3)
    : Op (dev, command), w1_ (w1), w2_ (w2), w3_ (w3)
{
  Pipeline::ShaderInfo shaderInfo = { 0, 3, 3, 16, 16, 1 };
  Pipeline::ConstantType act = { .i = 1 };
  shaderInfo.specialization_count = 1;
  pipeline0_.reset (new Pipeline (
      dev_, __get_matmul_shared_mem_comp_spv_code (),
      __get_matmul_shared_mem_comp_spv_size (), { act }, shaderInfo));
  shaderInfo.specialization_count = 0;
  pipeline1_.reset (
      new Pipeline (dev_, __get_matmul_shared_mem_comp_spv_code (),
                    __get_matmul_shared_mem_comp_spv_size (), {}, shaderInfo));
  pipeline2_.reset (
      new Pipeline (dev_, __get_matmul_shared_mem_comp_spv_code (),
                    __get_matmul_shared_mem_comp_spv_size (), {}, shaderInfo));

  shaderInfo.binding_count = 3;
  shaderInfo.push_constant_count = 1;
  shaderInfo.specialization_count = 1;
  shaderInfo.local_y = 1;

  Pipeline::ConstantType op_type = { .i = 2 };
  pipeline3_.reset (new Pipeline (dev_, __get_element_wise_comp_spv_code (),
                                  __get_element_wise_comp_spv_size (),
                                  { op_type }, shaderInfo));
}

VkResult
FeedForward::init () noexcept
{
  if (w3_.width () != w1_.width ())
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  auto ret = VK_SUCCESS;
  if ((ret = pipeline0_->init ()) != VK_SUCCESS)
    {
      return ret;
    }

  if ((ret = pipeline1_->init ()) != VK_SUCCESS)
    {
      return ret;
    }

  if ((ret = pipeline2_->init ()) != VK_SUCCESS)
    {
      return ret;
    }

  if ((ret = pipeline3_->init ()) != VK_SUCCESS)
    {
      return ret;
    }

  return VK_SUCCESS;
}

uint64_t
FeedForward::time () noexcept
{
  return std::max (pipeline0_->time (), pipeline2_->time ())
         + pipeline1_->time () + pipeline3_->time ();
}

VkResult
FeedForward::operator() (VkTensor X, VkTensor &output) noexcept
{
  t0_ = VkTensor (X.channels (), X.height (), w1_.width (), dev_,
                  VkTensor::FP32, false);
  t1_ = VkTensor (X.channels (), X.height (), w3_.width (), dev_,
                  VkTensor::FP32, false);
  t2_ = VkTensor::like (t0_);
  output = VkTensor (X.channels (), X.height (), w2_.width (), dev_);

  VkResult ret = VK_SUCCESS;
  if ((ret = t0_.create ()) != VK_SUCCESS)
    {
      return ret;
    }

  if ((ret = t1_.create ()) != VK_SUCCESS)
    {
      return ret;
    }

  if ((ret = t2_.create ()) != VK_SUCCESS)
    {
      return ret;
    }

  if ((ret = output.create ()) != VK_SUCCESS)
    {
      return ret;
    }

  Pipeline::ConstantType M = { .i = (int)X.height () };
  Pipeline::ConstantType N = { .i = (int)w1_.width () };
  Pipeline::ConstantType K = { .i = (int)X.width () };

  uint32_t groupx = (N.i + 31) / 32, groupy = (M.i + 31) / 32, groupz = 1;
  if ((ret = pipeline0_->set_group (groupx, groupy, groupz)) != VK_SUCCESS)
    {
      return ret;
    }

  if ((ret = pipeline2_->set_group (groupx, groupy, groupz)) != VK_SUCCESS)
    {
      return ret;
    }

  command_->record_pipeline (*pipeline0_, { X, w1_, t0_ }, { M, N, K });
  command_->record_pipeline (*pipeline2_, { X, w3_, t1_ }, { M, N, K });

  t0_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  t0_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  t1_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  t1_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  groupx = (t2_.channels () * t2_.height () * t2_.width () + 15) / 16;
  if ((ret = pipeline3_->set_group (groupx, 1, 1)) != VK_SUCCESS)
    {
      return ret;
    }

  Pipeline::ConstantType elems = { .i = static_cast<int> (t2_.size ()) };
  command_->record_pipeline (*pipeline3_, { t0_, t1_, t2_ }, { elems });
  t2_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  t2_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  M.i = t2_.height ();
  K.i = t2_.width ();
  N.i = output.width ();

  groupx = (N.i + 31) / 32;
  groupy = (M.i + 31) / 32;
  if ((ret = pipeline1_->set_group (groupx, groupy, groupz)) != VK_SUCCESS)
    {
      return ret;
    }

  command_->record_pipeline (*pipeline1_, { t2_, w2_, output }, { M, N, K });
  output.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  output.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  return VK_SUCCESS;
}
