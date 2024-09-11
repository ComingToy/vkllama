#ifndef __VKLLAMA_EMBEDDING_H__
#define __VKLLAMA_EMBEDDING_H__
#include "src/ops/op.h"
#include <memory>

namespace vkllama
{
class GPUDevice;
class Command;
class Embedding : public Op
{
public:
  Embedding (GPUDevice *dev, Command *, Tensor vocab, const uint32_t UNK,
             Tensor::DType dtype = Tensor::FP32);

  absl::Status init () noexcept override;
  uint64_t time () noexcept override;
  absl::StatusOr<Tensor> operator() (Tensor indices) noexcept;

private:
  std::unique_ptr<Pipeline> pipeline_;
  Tensor vocab_;
  const uint32_t UNK_;
  const Tensor::DType dtype_;
};

}
#endif
