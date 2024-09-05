#include "src/ops/update_kv_cache.h"
#include "src/core/command.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <array>
#include <memory>

namespace vkllama
{
UpdateKVCache::UpdateKVCache (GPUDevice *gpu, Command *command,
                              const VkTensor::DType dtype)
    : Op (gpu, command), dtype_ (dtype)
{
}

VkResult
UpdateKVCache::init () noexcept
{
  const auto *spv_code = dtype_ == VkTensor::FP32
                             ? nullptr
                             : __get_update_kvcache_fp16_comp_spv_code ();

  size_t spv_size = dtype_ == VkTensor::FP32
                        ? 0
                        : __get_update_kvcache_fp16_comp_spv_size ();

  Pipeline::ShaderInfo info = { 0, 2, sizeof (uint32_t) * 6, 16, 16, 1 };
  ShaderConstants specs;
  pipeline_
      = std::make_unique<Pipeline> (dev_, spv_code, spv_size, specs, info);
  return pipeline_->init ();
}

VkResult
UpdateKVCache::operator() (VkTensor cache, VkTensor key_or_value,
                           const std::array<size_t, 2> &offset) noexcept
{
  if (cache.height () < key_or_value.height () + offset[1]
      || cache.channels () < key_or_value.channels () + offset[0]
      || cache.width () != key_or_value.width ())
    {
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

  uint32_t flat_offset
      = static_cast<uint32_t> (offset[0] * cache.width () * cache.height ()
                               + offset[1] * cache.width ());

  ShaderConstants constants
      = { (uint32_t)key_or_value.channels (), (uint32_t)key_or_value.height (),
          (uint32_t)key_or_value.width (),    (uint32_t)cache.height (),
          (uint32_t)cache.width (),           (uint32_t)offset[1] };

  const uint32_t group_x = (key_or_value.width () + 15) / 16,
                 group_y = (key_or_value.height () + 15) / 16,
                 group_z = key_or_value.channels ();
  auto ret = pipeline_->set_group (group_x, group_y, group_z);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  ret = command_->record_pipeline (*pipeline_, { key_or_value, cache },
                                   constants);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  cache.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  cache.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return VK_SUCCESS;
}

uint64_t
UpdateKVCache::time () noexcept
{
  return pipeline_->time ();
}
}
