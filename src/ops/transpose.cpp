#include "src/ops/transpose.h"
#include "src/core/command.h"
#include "src/ops/op.h"
#include "src/shaders/vkllama_comp_shaders.h"

namespace vkllama
{
Transpose::Transpose (GPUDevice *gpu, Command *command, const int type,
                      VkTensor::DType const dtype)
    : Op (gpu, command), dtype_ (dtype), trans_type_ (type)
{
}

VkResult
Transpose::init () noexcept
{
  Pipeline::ShaderInfo info = { 0, 2, 6 * sizeof (uint32_t), 8, 4, 4 };
  const auto *spv_code = dtype_ == VkTensor::FP32
                             ? __get_transpose_type0_comp_spv_code ()
                             : __get_transpose_type0_fp16_comp_spv_code ();
  auto spv_size = dtype_ == VkTensor::FP32
                      ? __get_transpose_type0_comp_spv_size ()
                      : __get_transpose_type0_fp16_comp_spv_size ();

  pipeline_.reset (new Pipeline (dev_, spv_code, spv_size, {}, info));

  return pipeline_->init ();
}

VkResult
Transpose::operator() (VkTensor in, VkTensor &out) noexcept
{
  if (in.dtype () != dtype_)
    {
      return VK_ERROR_UNKNOWN;
    }

  if (trans_type_ != 0)
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  out = VkTensor (in.height (), in.channels (), in.width (), dev_,
                  in.dtype ());

  auto ret = out.create ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  uint32_t group_x = (out.width () + 7) / 8, group_y = (out.height () + 3) / 4,
           group_z = (out.channels () + 3) / 4;

  if ((ret = pipeline_->set_group (group_x, group_y, group_z)) != VK_SUCCESS)
    {
      return ret;
    }

  ShaderConstants shape
      = { (uint32_t)in.channels (), (uint32_t)in.height (),
          (uint32_t)in.width (),    (uint32_t)out.channels (),
          (uint32_t)out.height (),  (uint32_t)out.width () };
  if ((ret = command_->record_pipeline (*pipeline_, { in, out }, shape))
      != VK_SUCCESS)
    {
      return ret;
    }

  out.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return VK_SUCCESS;
};

uint64_t
Transpose::time () noexcept
{
  return pipeline_->time ();
}
}
