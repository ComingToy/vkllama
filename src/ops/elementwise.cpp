#include "src/ops/elementwise.h"
#include "src/core/command.h"
#include "src/core/gpu_device.h"
#include "src/shaders/vkllama_comp_shaders.h"

namespace vkllama
{
ElementWise::ElementWise (GPUDevice *dev, Command *command, const int type,
                          Tensor::DType dtype)
    : Op (dev, command), type_ (type), dtype_ (dtype)
{
}

absl::Status
ElementWise::init () noexcept
{
  Pipeline::ShaderInfo info = { 1, 3, sizeof (int), 128, 1, 1 };

  uint32_t bytes = sizeof (int) + sizeof (__vkllama_fp16_t) * 2;

  Pipeline::ShaderInfo info1 = { 1, 2, bytes, 128, 1, 1 };
  ShaderConstants constants = { type_ };

  const uint8_t *spv_code = nullptr;
  size_t spv_size = 0;

  if (dtype_ == Tensor::FP16 && dev_->support_16bit_storage ())
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
      return absl::InvalidArgumentError (
          "only fp16 dtype is supported on device");
    }

  pipeline0_.reset (new Pipeline (dev_, spv_code, spv_size, constants, info));

  auto ret = pipeline0_->init ();
  if (!ret.ok ())
    {
      return ret;
    }

  spv_code = dev_->support_fp16_arithmetic ()
                 ? __get_element_wise_constant_fp16a_comp_spv_code ()
                 : __get_element_wise_constant_fp16_comp_spv_code ();
  spv_size = dev_->support_fp16_arithmetic ()
                 ? __get_element_wise_constant_fp16a_comp_spv_size ()
                 : __get_element_wise_constant_fp16_comp_spv_size ();

  pipeline1_.reset (new Pipeline (dev_, spv_code, spv_size, constants, info1));
  return pipeline1_->init ();
}

uint64_t
ElementWise::time () noexcept
{
  return std::max (pipeline0_->time (), pipeline1_->time ());
}

absl::StatusOr<Tensor>
ElementWise::operator() (Tensor x, Tensor y) noexcept
{
  if (x.dtype () != y.dtype () || x.dtype () != dtype_)
    {
      return absl::InvalidArgumentError (
          "elementwise op: x.dtype() != y.dtype()");
    }

  if (x.channels () != y.channels () || x.height () != y.height ()
      || x.width () != y.width ())
    {
      return absl::InvalidArgumentError (
          absl::StrFormat ("elementwise op: shape error. shape of x = (%zu, "
                           "%zu, %zu), y = (%zu, %zu, %zu)",
                           x.channels (), x.height (), x.width (),
                           y.channels (), y.height (), y.width ()));
    }

  auto out = Tensor::like (x);

  auto ret = absl::OkStatus ();
  if (!(ret = out.create ()).ok ())
    {
      return ret;
    }

  ret = pipeline0_->set_group ((x.size () + 127) / 128, 1, 1);
  if (!ret.ok ())
    {
      return ret;
    }

  ShaderConstants constants = { static_cast<int> (x.size ()) };
  if (!(ret
        = command_->record_pipeline (*pipeline0_, { x, y, out }, constants))
           .ok ())
    {
      return ret;
    }

  out.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return out;
}

absl::StatusOr<Tensor>
ElementWise::operator() (Tensor x, float y) noexcept
{
  if (x.dtype () != dtype_)
    {
      return absl::OkStatus ();
    }

  auto out = Tensor::like (x);
  absl::Status ret;
  if (!(ret = out.create ()).ok ())
    {
      return ret;
    }

  ret = pipeline1_->set_group ((x.size () + 127) / 128, 1, 1);
  if (!ret.ok ())
    {
      return ret;
    }

  ShaderConstants constants = { (int)x.size () };
  if (!dev_->support_fp16_arithmetic ())
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
  if (!ret.ok ())
    {
      return ret;
    }

  out.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return out;
}
}

