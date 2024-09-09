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

absl::Status
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

absl::Status
UpdateKVCache::operator() (VkTensor cache, VkTensor key_or_value,
                           const uint32_t offset) noexcept
{
  if (cache.height () < key_or_value.height ()
      || cache.channels () < key_or_value.channels ()
      || cache.width () != key_or_value.width ())
    {
      return absl::OutOfRangeError (
          absl::StrFormat ("size of value is larger than cache. value.shape = "
                           "(%zu, %zu, %zu), cache.shape = (%zu, %zu, %zu)",
                           key_or_value.channels (), key_or_value.height (),
                           key_or_value.width (), cache.channels (),
                           cache.height (), cache.width ()));
    }

  ShaderConstants constants
      = { (uint32_t)key_or_value.channels (), (uint32_t)key_or_value.height (),
          (uint32_t)key_or_value.width (),    (uint32_t)cache.height (),
          (uint32_t)cache.width (),           offset };

  const uint32_t group_x = (key_or_value.width () + 15) / 16,
                 group_y = (key_or_value.height () + 15) / 16,
                 group_z = key_or_value.channels ();
  auto ret = pipeline_->set_group (group_x, group_y, group_z);
  if (!ret.ok ())
    {
      return ret;
    }

  ret = command_->record_pipeline (*pipeline_, { key_or_value, cache },
                                   constants);
  if (!ret.ok ())
    {
      return ret;
    }

  cache.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  cache.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return absl::OkStatus ();
}

uint64_t
UpdateKVCache::time () noexcept
{
  return pipeline_->time ();
}
}
