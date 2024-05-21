#include "rms_norm.h"
#include "src/core/command.h"
#include "src/core/pipeline.h"
#include "src/shaders/vkllama_comp_shaders.h"

RMSNorm::RMSNorm (GPUDevice *dev, Command *command, VkTensor weight,
                  const float eps_)
    : Op (dev, command), weight_ (weight)
{
  Pipeline::ConstantType power = { .f = 2.0f };
  Pipeline::ConstantType eps = { .f = eps_ };
  Pipeline::ShaderInfo info = { 2, 3, 3, 1, 32, 32 };
  pipeline_.reset (new Pipeline (dev_, __get_rms_norm_comp_spv_code (),
                                 __get_rms_norm_comp_spv_size (),
                                 { power, eps }, info));
}

VkResult
RMSNorm::init () noexcept
{
  auto ret = pipeline_->init ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  if ((ret = pipeline_->update_bindings ({ weight_ }, { 1 })) != VK_SUCCESS)
    {
      return ret;
    }
  return VK_SUCCESS;
}

uint64_t
RMSNorm::time () noexcept
{
  return pipeline_->time ();
}

VkResult
RMSNorm::operator() (VkTensor x, VkTensor &output) noexcept
{
  output = VkTensor (x.channels (), x.height (), x.width (), dev_,
                     VkTensor::FP32, false);
  auto ret = output.create ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  output.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  output.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  Pipeline::ConstantType C = { .u32 = (uint32_t)x.channels () };
  Pipeline::ConstantType H = { .u32 = (uint32_t)x.height () };
  Pipeline::ConstantType W = { .u32 = (uint32_t)x.width () };

  pipeline_->set_group (1, (H.u32 + 31) / 32, (C.u32 + 31) / 32);
  return command_->record_pipeline (*pipeline_, { x, output }, { 0, 2 },
                                    { C, H, W });
}
