#include "src/ops/embedding.h"
#include "src/core/command.h"
#include "src/core/common.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <memory>
#include <vector>

namespace vkllama
{
Embedding::Embedding (GPUDevice *dev, Command *command, Tensor vocab,
                      const uint32_t UNK, const Tensor::DType dtype)
    : Op (dev, command), vocab_ (vocab), UNK_ (UNK), dtype_ (dtype)
{
}

absl::Status
Embedding::init () noexcept
{

  if (dtype_ != FP16 && dtype_ != Q8_0)
    {
      return absl::InvalidArgumentError (
          "Embedding op: only fp16 and q8_0 are supported.");
    }

  if (dtype_ == FP16 && !dev_->support_16bit_storage ())
    {
      return absl::InvalidArgumentError ("fp16 is unsupported on device.");
    }

  if (dtype_ == Q8_0 && !dev_->support_8bit_storage ())
    {
      return absl::InvalidArgumentError ("q8_0 is unsupported on device.");
    }

  Pipeline::ShaderInfo info = { 1, 3, sizeof (ShapeConstant) * 2, 16, 2, 1 };
  ShaderConstants unk = { UNK_ };

  const auto *spv_code = __get_embedding_fp16_comp_spv_code ();
  auto spv_size = __get_embedding_fp16_comp_spv_size ();

  if (dtype_ == Q8_0)
    {
      spv_code = __get_embedding_q8_0_comp_spv_code ();
      spv_size = __get_embedding_q8_0_comp_spv_size ();
    }

  pipeline_.reset (new Pipeline (dev_, spv_code, spv_size, unk, info));
  auto ret = pipeline_->init ();
  if (!ret.ok ())
    {
      return ret;
    }

  return pipeline_->update_bindings ({ vocab_ }, { 0 });
}

absl::StatusOr<Tensor>
Embedding::operator() (Tensor indices) noexcept
{
  if (vocab_.channels () != 1 || indices.channels () != 1)
    {
      return absl::InvalidArgumentError (absl::StrFormat (
          "only channels = 1 is supported. vocab channels = %zu, indices "
          "channels = %zu\n",
          vocab_.channels (), indices.channels ()));
    }

  if (indices.dtype () != UINT32 || vocab_.dtype () != dtype_)
    {
      return absl::InvalidArgumentError (
          "only indices.dtype = UINT32 is supported");
    }

  if (out_.channels () != indices.height ()
      || out_.height () != indices.width ()
      || out_.width () != vocab_.width ())
    {
      out_ = Tensor (indices.height (), indices.width (), vocab_.width (),
                     dev_, FP16);
      VKLLAMA_STATUS_OK (out_.create ());
    }

  auto constants = vocab_.shape_constant () + indices.shape_constant ();

  uint32_t group_x = (indices.width () + 15) / 16,
           group_y = (indices.height () + 1) / 2;

  absl::Status ret;
  if (!(ret = pipeline_->set_group (group_x, group_y, 1)).ok ())
    {
      return ret;
    }

  ret = command_->record_pipeline (*pipeline_, { indices, out_ }, { 1, 2 },
                                   constants);
  if (!ret.ok ())
    {
      return ret;
    }

  out_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  static bool __enable_debug_log = false;
  if (__enable_debug_log)
    {
      VKLLAMA_STATUS_OK (
          command_->print_tensor_mean ("embedding outut mean: ", out_));
      __enable_debug_log = false;
    }

  return out_;
}

uint64_t
Embedding::time () noexcept
{
  return pipeline_->time ();
}
}

