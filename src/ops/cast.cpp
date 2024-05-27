#include "src/ops/cast.h"
#include "src/core/command.h"
#include "src/shaders/vkllama_comp_shaders.h"

Cast::Cast (GPUDevice *gpu, Command *command, const VkTensor::DType from,
            const VkTensor::DType to)
    : Op (gpu, command), from_ (from), to_ (to)
{
}

VkResult
Cast::init () noexcept
{
  const uint8_t *spv_code = nullptr;
  size_t spv_size = 0;
  if (from_ == VkTensor::FP32 && to_ == VkTensor::FP16)
    {
      spv_code = __get_cast_fp32_to_fp16_comp_spv_code ();
      spv_size = __get_cast_fp32_to_fp16_comp_spv_size ();
    }
  else if (from_ == VkTensor::FP16 && to_ == VkTensor::FP32)
    {
      spv_code = __get_cast_fp16_to_fp32_comp_spv_code ();
      spv_size = __get_cast_fp32_to_fp16_comp_spv_size ();
    }
  else
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  Pipeline::ShaderInfo info = { 0, 2, 1, 128, 1, 1 };
  pipeline_.reset (new Pipeline (dev_, spv_code, spv_size, {}, info));

  return pipeline_->init ();
}

uint64_t
Cast::time () noexcept
{
  return pipeline_->time ();
}

VkResult
Cast::operator() (VkTensor from, VkTensor &to) noexcept
{
  if (from.dtype () != from_)
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  to = VkTensor (from.channels (), from.height (), from.width (), dev_, to_);
  auto ret = to.create ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  ret = pipeline_->set_group ((from.size () + 127) / 128, 1, 1);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  Pipeline::ConstantType N = { .u32 = static_cast<uint32_t> (from.size ()) };
  ret = command_->record_pipeline (*pipeline_, { from, to }, { N });
  to.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  to.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  return VK_SUCCESS;
}
