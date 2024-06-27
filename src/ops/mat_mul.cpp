#include "mat_mul.h"
#include "src/core/command.h"
#include "src/core/gpu_device.h"
#include "src/core/pipeline.h"
#include "src/core/tensor.h"
#include "src/shaders/vkllama_comp_shaders.h"

namespace vkllama
{
MatMul::MatMul (GPUDevice *dev, Command *command, VkTensor weight,
                const float scale, const float bias, const int act,
                const int broadcast_type, const bool transpose_b,
                const VkTensor::DType dtype)
    : Op (dev, command), weight_ (weight), broadcast_type_ (broadcast_type),
      act_ (act), transpose_b_ (transpose_b), dtype_ (dtype), scale_ (scale),
      bias_ (bias)
{
}

MatMul::MatMul (GPUDevice *dev, Command *command, const float scale,
                const float bias, const int act, const int broadcast_type,
                const bool transpose_b, const VkTensor ::DType dtype)
    : Op (dev, command), broadcast_type_ (broadcast_type), act_ (act),
      transpose_b_ (transpose_b), dtype_ (dtype), scale_ (scale), bias_ (bias)
{
}

VkResult
MatMul::init () noexcept
{
  if (dtype_ == VkTensor::FP16 && !dev_->support_16bit_storage ())
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  Pipeline::ShaderInfo info = { 4, 3, 4 * sizeof (int), 16, 16, 1 };

  if (weight_.size () > 0 && weight_.dtype () != dtype_)
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  const uint8_t *pcode = nullptr;
  size_t code_size = 0;

  dev_->support_fp16_arithmetic ();
#define __SPV_SELECTOR(__boradcast)                                           \
  do                                                                          \
    {                                                                         \
      if (dtype_ == VkTensor::FP32)                                           \
        {                                                                     \
          pcode = __get_matmul_broadcast##__boradcast##_comp_spv_code ();     \
          code_size = __get_matmul_broadcast##__boradcast##_comp_spv_size (); \
        }                                                                     \
      else if (dtype_ == VkTensor::FP16 && dev_->support_fp16_arithmetic ())  \
        {                                                                     \
          pcode                                                               \
              = __get_matmul_broadcast##__boradcast##_fp16a_comp_spv_code (); \
          code_size                                                           \
              = __get_matmul_broadcast##__boradcast##_fp16a_comp_spv_size (); \
        }                                                                     \
      else if (dtype_ == VkTensor::FP16)                                      \
        {                                                                     \
          pcode                                                               \
              = __get_matmul_broadcast##__boradcast##_fp16_comp_spv_code ();  \
          code_size                                                           \
              = __get_matmul_broadcast##__boradcast##_fp16_comp_spv_size ();  \
        }                                                                     \
      else                                                                    \
        {                                                                     \
          return VK_ERROR_FORMAT_NOT_SUPPORTED;                               \
        }                                                                     \
    }                                                                         \
  while (0)

  if (broadcast_type_ == 0)
    {
      __SPV_SELECTOR (0);
    }
  else if (broadcast_type_ == 1)
    {
      __SPV_SELECTOR (1);
    }
  else if (broadcast_type_ == 2)
    {
      __SPV_SELECTOR (2);
    }
  else
    {
      return VK_ERROR_UNKNOWN;
    }

  pipeline_.reset (new Pipeline (dev_, pcode, code_size,
                                 { act_, (int)transpose_b_, scale_, bias_ },
                                 info));

  auto ret = pipeline_->init ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  if (weight_.size () == 0)
    return VK_SUCCESS;

  ret = pipeline_->update_bindings ({ weight_ }, { 1 });
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  return VK_SUCCESS;
}

uint64_t
MatMul::time () noexcept
{
  return pipeline_->time ();
}

VkResult
MatMul::operator() (VkTensor a, VkTensor &c) noexcept
{
  if (weight_.size () == 0 || a.dtype () != weight_.dtype ()
      || a.dtype () != dtype_)
    {
      return VK_ERROR_UNKNOWN;
    }

  if ((broadcast_type_ == 0 && weight_.channels () != a.channels ())
      || (broadcast_type_ == 1 && weight_.channels () != 1))
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  size_t out_h = a.height (),
         out_w = transpose_b_ ? weight_.height () : weight_.width ();
  c = VkTensor (std::max (a.channels (), weight_.channels ()), out_h, out_w,
                dev_, dtype_, false);

  auto ret = c.create ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  int channels = std::max (a.channels (), weight_.channels ());
  ShaderConstants constants
      = { channels, (int)a.height (), (int)out_w, (int)a.width () };

  uint32_t groupx = (out_w + 31) / 32, groupy = (a.height () + 31) / 32,
           groupz = channels;
  pipeline_->set_group (groupx, groupy, groupz);

  ret = command_->record_pipeline (*pipeline_, { a, c }, { 0, 2 }, constants);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  c.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  c.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return VK_SUCCESS;
}

VkResult
MatMul::operator() (VkTensor a, VkTensor b, VkTensor &c) noexcept
{
  if (b.size () == 0 || a.dtype () != b.dtype () || a.dtype () != dtype_)
    {
      return VK_ERROR_UNKNOWN;
    }

  if ((broadcast_type_ == 0 && b.channels () != a.channels ())
      || (broadcast_type_ == 1 && b.channels () != 1))
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  size_t out_h = a.height (), out_w = transpose_b_ ? b.height () : b.width ();
  c = VkTensor (std::max (a.channels (), b.channels ()), out_h, out_w, dev_,
                dtype_, false);

  auto ret = c.create ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  int channels = std::max (a.channels (), b.channels ());

  ShaderConstants constants
      = { channels, (int)a.height (), (int)out_w, (int)a.width () };

  uint32_t groupx = (out_w + 31) / 32, groupy = (a.height () + 31) / 32,
           groupz = channels;
  pipeline_->set_group (groupx, groupy, groupz);

  ret = command_->record_pipeline (*pipeline_, { a, b, c }, constants);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  c.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  c.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return VK_SUCCESS;
}
}

