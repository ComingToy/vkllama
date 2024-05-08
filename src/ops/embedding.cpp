#include "src/ops/embedding.h"
#include "src/core/command.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <memory>
#include <vector>

Embedding::Embedding (GPUDevice *dev, Command *command, const uint32_t UNK)
    : Op (dev, command), UNK_ (UNK)
{
}

VkResult
Embedding::init () noexcept
{
  Pipeline::ShaderInfo info = { 1, 3, 4, 16, 16, 1 };
  Pipeline::ConstantType unk = { .u32 = UNK_ };
  pipeline_.reset (new Pipeline (dev_, __get_embedding_comp_spv_code (),
                                 __get_embedding_comp_spv_size (), { unk },
                                 info));
  return pipeline_->init ();
}

VkResult
Embedding::operator() (VkTensor vocab, VkTensor indices,
                       VkTensor &out) noexcept
{
  if (vocab.channels () != 1 || indices.channels () != 1
      || indices.dtype () != VkTensor::UINT32
      || vocab.dtype () != VkTensor::FP32)
    {
      return VK_ERROR_UNKNOWN;
    }

  out = VkTensor (indices.height (), indices.width (), vocab.width (), dev_,
                  vocab.dtype ());
  auto ret = out.create ();

  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  std::vector<Pipeline::ConstantType> constants
      = { { .u32 = (uint32_t)indices.height () },
          { .u32 = (uint32_t)indices.width () },
          { .u32 = (uint32_t)vocab.height () },
          { .u32 = (uint32_t)vocab.width () } };

  uint32_t group_x = (indices.width () + 15) / 16,
           group_y = (indices.height () + 15) / 16;

  if ((ret = pipeline_->set_group (group_x, group_y, 1)) != VK_SUCCESS)
    {
      return ret;
    }

  ret = command_->record_pipeline (*pipeline_, { vocab, indices, out },
                                   constants);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  out.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out.set_pipeline_stage (VK_SHADER_STAGE_COMPUTE_BIT);

  return VK_SUCCESS;
}

uint64_t
Embedding::time () noexcept
{
  return pipeline_->time ();
}
