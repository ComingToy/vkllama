#ifndef __VKLLAMA_UPDATE_KV_CACHE_H__
#define __VKLLAMA_UPDATE_KV_CACHE_H__

#include "src/core/pipeline.h"
#include "src/ops/op.h"
#include <array>
#include <memory>
namespace vkllama
{
class UpdateKVCache : public Op
{
public:
  UpdateKVCache (GPUDevice *gpu, Command *command,
                 const VkTensor::DType dtype);

  VkResult init () noexcept override;
  uint64_t time () noexcept override;
  VkResult operator() (VkTensor cache, VkTensor key_or_value,
                       std::array<size_t, 2> const &offset) noexcept;

private:
  std::unique_ptr<Pipeline> pipeline_;
  VkTensor::DType dtype_;
};
}

#endif
