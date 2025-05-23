#include "mat_mul.h"
#include "src/core/command.h"
#include "src/core/common.h"
#include "src/core/gpu_device.h"
#include "src/core/pipeline.h"
#include "src/core/tensor.h"
#include "src/shaders/matmul_conf.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <cstdio>

namespace vkllama
{
MatMul::MatMul (GPUDevice *dev, Command *command, Tensor weight,
                const float scale, const float bias, const int act,
                const int broadcast_type, const bool transpose_b,
                const Tensor::DType a_dtype, const Tensor::DType b_dtype)
    : Op (dev, command), weight_ (weight), broadcast_type_ (broadcast_type),
      act_ (act), transpose_b_ (transpose_b), a_dtype_ (a_dtype),
      b_dtype_ (b_dtype), scale_ (scale), bias_ (bias)
{
}

MatMul::MatMul (GPUDevice *dev, Command *command, const float scale,
                const float bias, const int act, const int broadcast_type,
                const bool transpose_b, const Tensor::DType a_dtype,
                const Tensor::DType b_dtype)
    : Op (dev, command), broadcast_type_ (broadcast_type), act_ (act),
      transpose_b_ (transpose_b), a_dtype_ (a_dtype), b_dtype_ (b_dtype),
      scale_ (scale), bias_ (bias)
{
}

absl::Status
MatMul::init () noexcept
{
  if ((a_dtype_ == FP16 || b_dtype_ == FP16)
      && !dev_->support_16bit_storage ())
    {
      return absl::InvalidArgumentError ("fp16 is unsupported on device");
    }

  Pipeline::ShaderInfo info
      = { 4, 3, 3 * sizeof (ShapeConstant), (uint32_t)dev_->subgroup_size (),
          1, 1 };

  if (weight_.size () > 0 && weight_.dtype () != b_dtype_)
    {
      return absl::InvalidArgumentError (absl::StrFormat (
          "matmul operator defined as %d b_dtype but dtype of "
          "weight tensor is %d",
          int (b_dtype_), int (weight_.dtype ())));
    }

  // if (a_dtype_ == FP16 && b_dtype_ == Q8_0 && broadcast_type_ == 0
  //     && transpose_b_)
  //   {
  //     info.local_y = Q8_0_TILE_X_SIZE;
  //   }
  // else if (a_dtype_ == FP16 && b_dtype_ == FP16 && transpose_b_
  //          && broadcast_type_ == 0)
  //   {
  //     info.local_y = FP16_TILE_X_SIZE;
  //   }

  const uint8_t *pcode = nullptr;
  size_t code_size = 0;

#define __SPV_SELECTOR(__boradcast)                                              \
  do                                                                             \
    {                                                                            \
      if (a_dtype_ == FP16 && b_dtype_ == FP16 && transpose_b_                   \
          && dev_->support_fp16_arithmetic () && broadcast_type_ == 0)           \
        {                                                                        \
          pcode = __get_matmul_b0_tb_fp16a_v2_comp_spv_code ();                  \
          code_size = __get_matmul_b0_tb_fp16a_v2_comp_spv_size ();              \
        }                                                                        \
      else if (a_dtype_ == FP16 && b_dtype_ == FP16 && transpose_b_              \
               && broadcast_type_ == 0)                                          \
        {                                                                        \
          pcode = __get_matmul_b0_tb_fp16_v2_comp_spv_code ();                   \
          code_size = __get_matmul_b0_tb_fp16_v2_comp_spv_size ();               \
        }                                                                        \
      else if (a_dtype_ == b_dtype_ && a_dtype_ == FP16                          \
               && dev_->support_fp16_arithmetic ())                              \
        {                                                                        \
          pcode                                                                  \
              = __get_matmul_broadcast##__boradcast##_fp16a_v2_comp_spv_code (); \
          code_size                                                              \
              = __get_matmul_broadcast##__boradcast##_fp16a_v2_comp_spv_size (); \
        }                                                                        \
      else if (a_dtype_ == FP16 && b_dtype_ == FP16)                             \
        {                                                                        \
          pcode                                                                  \
              = __get_matmul_broadcast##__boradcast##_fp16_v2_comp_spv_code ();  \
          code_size                                                              \
              = __get_matmul_broadcast##__boradcast##_fp16_v2_comp_spv_size ();  \
        }                                                                        \
      else if (a_dtype_ == FP16 && b_dtype_ == Q8_0 && transpose_b_              \
               && broadcast_type_ == 0)                                          \
        {                                                                        \
          pcode = __get_matmul_b0_fp16_x_q8_0_comp_spv_code ();                  \
          code_size = __get_matmul_b0_fp16_x_q8_0_comp_spv_size ();              \
        }                                                                        \
      else                                                                       \
        {                                                                        \
          return absl::InvalidArgumentError (                                    \
              absl::StrFormat ("%d a_dtype and %d b_dtype is unsupported",       \
                               int (a_dtype_), int (b_dtype_)));                 \
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

absl::StatusOr<Tensor>
MatMul::operator() (Tensor a) noexcept
{
  if (weight_.size () == 0 || a.dtype () != a_dtype_)
    {
      return absl::InvalidArgumentError (
          absl::StrFormat ("matmul op defined with %d a dtype but %d given.",
                           int (a_dtype_), int (a.dtype ())));
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

  size_t out_c = std::max (a.channels (), weight_.channels ()),
         out_h = a.height (),
         out_w = transpose_b_ ? weight_.height () : weight_.width ();

  if (out_.channels () != out_c || out_.height () != out_h
      || out_.width () != out_w)
    {
      out_ = Tensor (out_c, out_h, out_w, dev_, FP16, false);
      VKLLAMA_STATUS_OK (out_.create ());
    }

  int channels = std::max (a.channels (), weight_.channels ());

  uint32_t groupx = out_w, groupy = a.height (), groupz = channels;
  if (a_dtype_ == FP16 && b_dtype_ == Q8_0 && broadcast_type_ == 0
      && transpose_b_)
    {
      groupx = (out_w + Q8_0_TILE_X_SIZE - 1) / Q8_0_TILE_X_SIZE;
    }
  else if (a_dtype_ == FP16 && b_dtype_ == FP16 && transpose_b_
           && broadcast_type_ == 0)
    {
      groupx = (out_w + FP16_TILE_X_SIZE - 1) / FP16_TILE_X_SIZE;
    }

  if (auto ret = pipeline_->set_group (groupx, groupy, groupz); !ret.ok ())
    {
      return ret;
    }

  ShaderConstants constants = a.shape_constant () + weight_.shape_constant ()
                              + out_.shape_constant ();

  auto ret = command_->record_pipeline (*pipeline_, { a, out_ }, { 0, 2 },
                                        constants);
  if (!ret.ok ())
    {
      return ret;
    }

  out_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return out_;
}

absl::StatusOr<Tensor>
MatMul::operator() (Tensor a, Tensor b) noexcept
{
  if (b.size () == 0 || a.dtype () != a_dtype_ || b.dtype () != b_dtype_)
    {
      return absl::InvalidArgumentError (absl::StrFormat (
          "matmul op defined with %d a_dtype and %d b_dtype but inputs "
          "a.dtype() = %d, b.dtype() = %d",
          int (a_dtype_), int (b_dtype_), int (a.dtype ()), int (b.dtype ())));
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
  size_t out_c = std::max (a.channels (), b.channels ());

  if (out_.channels () != out_c || out_.height () != out_h
      || out_.width () != out_w)
    {
      out_ = Tensor (out_c, out_h, out_w, dev_, FP16, false);
      VKLLAMA_STATUS_OK (out_.create ());
    }

  int channels = std::max (a.channels (), b.channels ());

  uint32_t groupx = out_w, groupy = a.height (), groupz = channels;

  if (a_dtype_ == FP16 && b_dtype_ == Q8_0 && transpose_b_
      && broadcast_type_ == 0)
    {
      groupx = (out_w + Q8_0_TILE_X_SIZE - 1) / Q8_0_TILE_X_SIZE;
    }
  else if (a_dtype_ == FP16 && b_dtype_ == FP16 && transpose_b_
           && broadcast_type_ == 0)
    {
      groupx = (out_w + FP16_TILE_X_SIZE - 1) / FP16_TILE_X_SIZE;
    }

  auto s = pipeline_->set_group (groupx, groupy, groupz);
  if (!s.ok ())
    return s;

  auto constants
      = a.shape_constant () + b.shape_constant () + out_.shape_constant ();

  auto ret = command_->record_pipeline (*pipeline_, { a, b, out_ }, constants);
  if (!ret.ok ())
    {
      return ret;
    }

  out_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return out_;
}
}

