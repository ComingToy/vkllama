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

  uint32_t bytes = sizeof (int)
                   + (dtype_ == VkTensor::FP16 ? sizeof (__vkllama_fp16_t) * 2
                                               : sizeof (float));
  Pipeline::ShaderInfo info1 = { 1, 3, bytes, 128, 1, 1 };
  ShaderConstants constants = { type_ };

  const uint8_t *spv_code = nullptr;
  size_t spv_size = 0;

  if (dtype_ == VkTensor::FP32)
    {
      spv_code = __get_element_wise_comp_spv_code ();
      spv_size = __get_element_wise_comp_spv_size ();
    }
  else if (dtype_ == VkTensor::FP16 && dev_->support_16bit_storage ())
    {
      spv_code = dev_->support_fp16_arithmetic ()
                     ? __get_element_wise_fp16a_comp_spv_code ()
                     : __get_element_wise_fp16_comp_spv_code ();
      spv_size = dev_->support_fp16_arithmetic ()
                     ? __get_element_wise_fp16a_comp_spv_size ()
                     : __get_element_wise_fp16_comp_spv_size ();
    }
  else
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  pipeline0_.reset (new Pipeline (dev_, spv_code, spv_size, constants, info));

  auto ret = pipeline0_->init ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  if (dtype_ == VkTensor::FP32)
    {
      spv_code = __get_element_wise_constant_comp_spv_code ();
      spv_size = __get_element_wise_constant_comp_spv_size ();
    }
  else if (dtype_ == VkTensor::FP16 && dev_->support_16bit_storage ())
    {
      spv_code = dev_->support_fp16_arithmetic ()
                     ? __get_element_wise_constant_fp16a_comp_spv_code ()
                     : __get_element_wise_constant_fp16_comp_spv_code ();
      spv_size = dev_->support_fp16_arithmetic ()
                     ? __get_element_wise_constant_fp16a_comp_spv_size ()
                     : __get_element_wise_constant_fp16_comp_spv_size ();
    }
  else
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

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

  ShaderConstants constants = { (int)x.size () };
  if (dtype_ == VkTensor::FP32 || !dev_->support_fp16_arithmetic ())
    {
      constants.push_back (y);
    }
  else
    {
      // constants.push_back (y);
      constants.push_back (__fp32_to_fp16 (y));
      constants.push_back (__fp32_to_fp16 (0)); // padding
    }
  ret = command_->record_pipeline (*pipeline1_, { x, out }, constants);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  out.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return VK_SUCCESS;
}
