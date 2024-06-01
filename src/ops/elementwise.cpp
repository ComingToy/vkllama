#include "src/ops/elementwise.h"
#include "src/core/command.h"
#include "src/core/gpu_device.h"
#include "src/shaders/vkllama_comp_shaders.h"

ElementWise::ElementWise (GPUDevice *dev, Command *command, const int type,
                          VkTensor::DType dtype)
    : Op (dev, command), type_ (type), dtype_ (dtype)
{
}

VkResult
ElementWise::init () noexcept
{
  Pipeline::ShaderInfo info = { 1, 3, sizeof (int), 128, 1, 1 };
  Pipeline::ShaderInfo info1
      = { 1, 3, sizeof (int) + sizeof (float), 128, 1, 1 };
  ShaderConstants constants = { type_ };

  const uint8_t *spv_code = dtype_ == VkTensor::FP16
                                ? __get_element_wise_fp16_comp_spv_code ()
                                : __get_element_wise_comp_spv_code ();

  size_t spv_size = dtype_ == VkTensor::FP16
                        ? __get_element_wise_fp16_comp_spv_size ()
                        : __get_element_wise_comp_spv_size ();

  pipeline0_.reset (new Pipeline (dev_, spv_code, spv_size, constants, info));

  auto ret = pipeline0_->init ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  spv_code = dtype_ == VkTensor::FP32
                 ? __get_element_wise_constant_comp_spv_code ()
                 : __get_element_wise_constant_fp16_comp_spv_code ();
  spv_size = dtype_ == VkTensor::FP32
                 ? __get_element_wise_constant_comp_spv_size ()
                 : __get_element_wise_constant_fp16_comp_spv_size ();

  pipeline1_.reset (new Pipeline (dev_, spv_code, spv_size, constants, info1));
  return pipeline1_->init ();
}

uint64_t
ElementWise::time () noexcept
{
  return pipeline0_->time ();
}

VkResult
ElementWise::operator() (VkTensor x, VkTensor y, VkTensor &out) noexcept
{
  if (x.dtype () != y.dtype () || x.dtype () != dtype_)
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  if (x.channels () != y.channels () || x.height () != y.height ()
      || x.width () != y.width ())
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  out = VkTensor::like (x);
  VkResult ret = VK_SUCCESS;
  if ((ret = out.create ()) != VK_SUCCESS)
    {
      return ret;
    }

  ret = pipeline0_->set_group ((x.size () + 127) / 128, 1, 1);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  ShaderConstants constants = { static_cast<int> (x.size ()) };
  if ((ret = command_->record_pipeline (*pipeline0_, { x, y, out }, constants))
      != VK_SUCCESS)
    {
      return ret;
    }

  out.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return VK_SUCCESS;
}

VkResult
ElementWise::operator() (VkTensor x, float y, VkTensor &out) noexcept
{
  if (x.dtype () != dtype_)
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  out = VkTensor::like (x);
  VkResult ret = VK_SUCCESS;
  if ((ret = out.create ()) != VK_SUCCESS)
    {
      return ret;
    }

  ret = pipeline1_->set_group ((x.size () + 127) / 128, 1, 1);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  ShaderConstants constants (static_cast<int> (x.size ()), y);
  ret = command_->record_pipeline (*pipeline1_, { x, out }, constants);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  out.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return VK_SUCCESS;
}
