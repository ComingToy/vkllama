#ifndef __VKLLAMA_EMBEDDING_H__
#define __VKLLAMA_EMBEDDING_H__
#include "src/ops/op.h"
#include <memory>

class GPUDevice;
class Command;
class Embedding : public Op
{
public:
  Embedding (GPUDevice *dev, Command *, VkTensor vocab, const uint32_t UNK,
             VkTensor::DType dtype = VkTensor::FP32);

  VkResult init () noexcept override;
  uint64_t time () noexcept override;
  VkResult operator() (VkTensor indices, VkTensor &out) noexcept;

private:
  std::unique_ptr<Pipeline> pipeline_;
  VkTensor vocab_;
  const uint32_t UNK_;
  const VkTensor::DType dtype_;
};
#endif
