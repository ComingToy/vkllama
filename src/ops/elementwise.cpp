#include "src/ops/elementwise.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include "src/core/command.h"
#include "src/core/gpu_device.h"

ElementWise::ElementWise (GPUDevice *dev, Command *command, const int type)
    : Op (dev, command), type_ (type)
{
}

VkResult
ElementWise::init () noexcept
{
  Pipeline::ShaderInfo info = { 1, 3, 1, 128, 1, 1 };
  Pipeline::ConstantType op_type = { .i = type_ };

  pipeline0_.reset (new Pipeline (dev_, __get_element_wise_comp_spv_code (),
                                  __get_element_wise_comp_spv_size (),
                                  { op_type }, info));

  auto ret = pipeline0_->init ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  info.push_constant_count = 2;
  info.binding_count = 2;
  pipeline1_.reset (new Pipeline (
      dev_, __get_element_wise_constant_comp_spv_code (),
      __get_element_wise_constant_comp_spv_size (), { op_type }, info));
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

  Pipeline::ConstantType n = { .i = static_cast<int> (x.size ()) };
  if ((ret = command_->record_pipeline (*pipeline0_, { x, y, out }, { n }))
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

  Pipeline::ConstantType n = { .i = static_cast<int> (x.size ()) };
  Pipeline::ConstantType alpha = { .f = y };
  ret = command_->record_pipeline (*pipeline1_, { x, out }, { n, alpha });
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  out.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return VK_SUCCESS;
}
