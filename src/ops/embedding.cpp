#include "src/ops/embedding.h"
#include "src/core/command.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <memory>
#include <vector>

Embedding::Embedding (GPUDevice *dev, Command *command, VkTensor vocab,
                      const uint32_t UNK, const VkTensor::DType dtype)
    : Op (dev, command), vocab_ (vocab), UNK_ (UNK), dtype_ (dtype)
{
}

VkResult
Embedding::init () noexcept
{
  if (dtype_ == VkTensor::FP16 && !dev_->support_16bit_storage ())
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  Pipeline::ShaderInfo info = { 1, 3, sizeof (uint32_t) * 4, 16, 16, 1 };
  ShaderConstants unk = { UNK_ };

  const auto *spv_code = dtype_ == VkTensor::FP32
                             ? __get_embedding_comp_spv_code ()
                             : __get_embedding_fp16_comp_spv_code ();

  const auto spv_size = dtype_ == VkTensor::FP32
                            ? __get_embedding_comp_spv_size ()
                            : __get_embedding_fp16_comp_spv_size ();

  pipeline_.reset (new Pipeline (dev_, spv_code, spv_size, unk, info));
  auto ret = pipeline_->init ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  return pipeline_->update_bindings ({ vocab_ }, { 0 });
}

VkResult
Embedding::operator() (VkTensor indices, VkTensor &out) noexcept
{
  if (vocab_.channels () != 1 || indices.channels () != 1
      || indices.dtype () != VkTensor::UINT32 || vocab_.dtype () != dtype_)
    {
      return VK_ERROR_UNKNOWN;
    }

  out = VkTensor (indices.height (), indices.width (), vocab_.width (), dev_,
                  vocab_.dtype ());
  auto ret = out.create ();

  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  ShaderConstants constants
      = { (uint32_t)indices.height (), (uint32_t)indices.width (),
          (uint32_t)vocab_.height (), (uint32_t)vocab_.width () };

  uint32_t group_x = (indices.width () + 15) / 16,
           group_y = (indices.height () + 15) / 16;

  if ((ret = pipeline_->set_group (group_x, group_y, 1)) != VK_SUCCESS)
    {
      return ret;
    }

  ret = command_->record_pipeline (*pipeline_, { indices, out }, { 1, 2 },
                                   constants);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  out.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  return VK_SUCCESS;
}

uint64_t
Embedding::time () noexcept
{
  return pipeline_->time ();
}
