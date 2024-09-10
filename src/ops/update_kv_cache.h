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
                 const Tensor::DType dtype);

  absl::Status init () noexcept override;
  uint64_t time () noexcept override;
  absl::Status operator() (Tensor cache, Tensor key_or_value,
                       uint32_t offset) noexcept;

private:
  std::unique_ptr<Pipeline> pipeline_;
  Tensor::DType dtype_;
};
}

#endif
