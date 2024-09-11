#ifndef __VKLLAMA_READ_KVCACHE_OP__
#define __VKLLAMA_READ_KVCACHE_OP__
#include "src/core/pipeline.h"
#include "src/ops/op.h"
#include <array>
#include <memory>

namespace vkllama
{
class ReadKVCache : public Op
{
public:
  ReadKVCache (GPUDevice *gpu, Command *command);
  absl::Status init () noexcept override;
  uint64_t time () noexcept override;
  absl::StatusOr<Tensor> operator() (Tensor cache, uint32_t offset,
                                     uint32_t len) noexcept;

private:
  std::unique_ptr<Pipeline> pipeline_;
};
};
#endif
