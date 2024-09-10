#include "mat_mul.h"
#include "src/core/command.h"
#include "src/core/gpu_device.h"
#include "src/core/pipeline.h"
#include "src/core/tensor.h"
#include "src/shaders/vkllama_comp_shaders.h"

namespace vkllama
{
MatMul::MatMul (GPUDevice *dev, Command *command, Tensor weight,
                const float scale, const float bias, const int act,
                const int broadcast_type, const bool transpose_b,
                const Tensor::DType dtype)
    : Op (dev, command), weight_ (weight), broadcast_type_ (broadcast_type),
      act_ (act), transpose_b_ (transpose_b), dtype_ (dtype), scale_ (scale),
      bias_ (bias)
{
}

MatMul::MatMul (GPUDevice *dev, Command *command, const float scale,
                const float bias, const int act, const int broadcast_type,
                const bool transpose_b, const Tensor ::DType dtype)
    : Op (dev, command), broadcast_type_ (broadcast_type), act_ (act),
      transpose_b_ (transpose_b), dtype_ (dtype), scale_ (scale), bias_ (bias)
{
}

absl::Status
MatMul::init () noexcept
{
  if (dtype_ == Tensor::FP16 && !dev_->support_16bit_storage ())
    {
      return absl::InvalidArgumentError ("fp16 is unsupported on device");
    }

  Pipeline::ShaderInfo info = { 4, 3, 4 * sizeof (int), 8, 16, 1 };
  if (dtype_ == Tensor::FP16)
    {
      info
          = { 4, 3, 4 * sizeof (int), (uint32_t)dev_->subgroup_size (), 8, 1 };
    }

  if (weight_.size () > 0 && weight_.dtype () != dtype_)
    {
      return absl::InvalidArgumentError (
          absl::StrFormat ("matmul operator defined as %d dtype but dtype of "
                           "weight tensor is %d",
                           int (dtype_), int (weight_.dtype ())));
    }

  const uint8_t *pcode = nullptr;
  size_t code_size = 0;

  dev_->support_fp16_arithmetic ();
#define __SPV_SELECTOR(__boradcast)                                              \
  do                                                                             \
    {                                                                            \
      if (dtype_ == Tensor::FP32)                                              \
        {                                                                        \
          pcode = __get_matmul_broadcast##__boradcast##_comp_spv_code ();        \
          code_size = __get_matmul_broadcast##__boradcast##_comp_spv_size ();    \
        }                                                                        \
      else if (dtype_ == Tensor::FP16 && dev_->support_fp16_arithmetic ())     \
        {                                                                        \
          pcode                                                                  \
              = __get_matmul_broadcast##__boradcast##_fp16a_v2_comp_spv_code (); \
          code_size                                                              \
              = __get_matmul_broadcast##__boradcast##_fp16a_v2_comp_spv_size (); \
        }                                                                        \
      else if (dtype_ == Tensor::FP16)                                         \
        {                                                                        \
          pcode                                                                  \
              = __get_matmul_broadcast##__boradcast##_fp16_v2_comp_spv_code ();  \
          code_size                                                              \
              = __get_matmul_broadcast##__boradcast##_fp16_v2_comp_spv_size ();  \
        }                                                                        \
      else                                                                       \
        {                                                                        \
          return absl::InvalidArgumentError (                                    \
              absl::StrFormat ("%d dtype is unsupported", int (dtype_)));        \
        }                                                                        \
    }                                                                            \
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
      return absl::InvalidArgumentError (absl::StrFormat (
          "broadcast_type %d is unsupported.", broadcast_type_));
    }

  pipeline_.reset (new Pipeline (dev_, pcode, code_size,
                                 { act_, (int)transpose_b_, scale_, bias_ },
                                 info));

  auto ret = pipeline_->init ();
  if (!ret.ok ())
    {
      return ret;
    }

  if (weight_.size () == 0)
    return absl::OkStatus ();

  ret = pipeline_->update_bindings ({ weight_ }, { 1 });
  if (!ret.ok ())
    {
      return ret;
    }

  return absl::OkStatus ();
}

uint64_t
MatMul::time () noexcept
{
  return pipeline_->time ();
}

absl::Status
MatMul::operator() (Tensor a, Tensor &c) noexcept
{
  if (weight_.size () == 0 || a.dtype () != weight_.dtype ()
      || a.dtype () != dtype_)
    {
      return absl::InvalidArgumentError (
          absl::StrFormat ("matmul op defined with %d dtype but %d given.",
                           int (dtype_), int (a.dtype ())));
    }

  if (broadcast_type_ == 0 && weight_.channels () != a.channels ())
    {
      return absl::InvalidArgumentError (absl::StrFormat (
          "matmul op defined with broadcast_type = %d but weight channels = "
          "%zu != a.channels = %zu",
          broadcast_type_, weight_.channels (), a.channels ()));
    }

  if (broadcast_type_ == 1 && weight_.channels () != 1)
    {
      return absl::InvalidArgumentError (
          absl::StrFormat ("matmul op defined with broadcast_type = %d but "
                           "weight channels = %zu != 1",
                           broadcast_type_, weight_.channels ()));
    }

  size_t out_h = a.height (),
         out_w = transpose_b_ ? weight_.height () : weight_.width ();
  c = Tensor (std::max (a.channels (), weight_.channels ()), out_h, out_w,
                dev_, dtype_, false);

  auto ret = c.create ();
  if (!ret.ok ())
    {
      return ret;
    }

  int channels = std::max (a.channels (), weight_.channels ());
  ShaderConstants constants
      = { channels, (int)a.height (), (int)out_w, (int)a.width () };

  if (dtype_ == Tensor::FP16)
    {
      uint32_t groupx = out_w, groupy = (a.height () + 7) / 8,
               groupz = channels;
      if (auto ret = pipeline_->set_group (groupx, groupy, groupz); !ret.ok ())
        {
          return ret;
        }
    }
  else
    {
      uint32_t groupx = (out_w + 31) / 32, groupy = (a.height () + 31) / 32,
               groupz = channels;
      if (auto ret = pipeline_->set_group (groupx, groupy, groupz); !ret.ok ())
        {
          return ret;
        }
    }

  ret = command_->record_pipeline (*pipeline_, { a, c }, { 0, 2 }, constants);
  if (!ret.ok ())
    {
      return ret;
    }

  c.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  c.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return absl::OkStatus ();
}

absl::Status
MatMul::operator() (Tensor a, Tensor b, Tensor &c) noexcept
{
  if (b.size () == 0 || a.dtype () != b.dtype () || a.dtype () != dtype_)
    {
      return absl::InvalidArgumentError (
          absl::StrFormat ("matmul op defined with %d dtype but inputs "
                           "a.dtype() = %d, b.dtype() = %d",
                           int (dtype_), int (a.dtype ()), int (b.dtype ())));
    }

  if ((broadcast_type_ == 0 && b.channels () != a.channels ())
      || (broadcast_type_ == 1 && b.channels () != 1))
    {
      return absl::InvalidArgumentError (
          absl::StrFormat ("matmul defined with %d broadcast_type but "
                           "a.channels() = %zu, b.channels() = %zu",
                           broadcast_type_, a.channels (), b.channels ()));
    }

  size_t out_h = a.height (), out_w = transpose_b_ ? b.height () : b.width ();
  c = Tensor (std::max (a.channels (), b.channels ()), out_h, out_w, dev_,
                dtype_, false);

  auto ret = c.create ();
  if (!ret.ok ())
    {
      return ret;
    }

  int channels = std::max (a.channels (), b.channels ());

  ShaderConstants constants
      = { channels, (int)a.height (), (int)out_w, (int)a.width () };

  if (dtype_ == Tensor::FP16)
    {
      uint32_t groupx = out_w, groupy = (a.height () + 7) / 8,
               groupz = channels;
      auto s = pipeline_->set_group (groupx, groupy, groupz);
      if (!s.ok ())
        return s;
    }
  else
    {
      uint32_t groupx = (out_w + 31) / 32, groupy = (a.height () + 31) / 32,
               groupz = channels;
      auto s = pipeline_->set_group (groupx, groupy, groupz);
      if (!s.ok ())
        {
          return s;
        }
    }

  ret = command_->record_pipeline (*pipeline_, { a, b, c }, constants);
  if (!ret.ok ())
    {
      return ret;
    }

  c.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  c.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return absl::OkStatus ();
}
}

