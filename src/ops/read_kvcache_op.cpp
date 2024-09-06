#include "src/ops/read_kvcache_op.h"
#include "src/core/command.h"
#include "src/shaders/vkllama_comp_shaders.h"

namespace vkllama
{

ReadKVCache::ReadKVCache (GPUDevice *gpu, Command *command) : Op (gpu, command)
{
}

VkResult
ReadKVCache::init () noexcept
{
  auto spv = __get_read_kvcache_fp16_comp_spv_code ();
  auto spv_size = __get_read_kvcache_fp16_comp_spv_size ();
  Pipeline::ShaderInfo info = { 0, 2, sizeof (uint32_t) * 5, 16, 16, 1 };

  pipeline_.reset (new Pipeline (dev_, spv, spv_size, {}, info));
  return pipeline_->init ();
}

uint64_t
ReadKVCache::time () noexcept
{
  return pipeline_->time ();
}

VkResult
ReadKVCache::operator() (VkTensor cache, uint32_t offset, uint32_t len,
                         VkTensor &key_or_value) noexcept
{
  if (len > cache.height ())
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  key_or_value = VkTensor (cache.channels (), len, cache.width (), dev_,
                           cache.dtype (), false);

  if (auto ret = key_or_value.create (); ret != VK_SUCCESS)
    {
      return ret;
    }

  uint32_t groupx = (key_or_value.width () + 15) / 16;
  uint32_t groupy = (key_or_value.height () + 15) / 16;
  uint32_t groupz = key_or_value.channels ();

  if (auto ret = pipeline_->set_group (groupx, groupy, groupz);
      ret != VK_SUCCESS)
    {
      return ret;
    }

  ShaderConstants constants
      = { (uint32_t)cache.channels (), (uint32_t)cache.height (),
          (uint32_t)cache.width (), offset, len };

  auto ret = command_->record_pipeline (*pipeline_, { cache, key_or_value },
                                        constants);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  key_or_value.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  key_or_value.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  return VK_SUCCESS;
}
};
