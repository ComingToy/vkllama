#ifndef __VKLLAMA_EMBEDDING_H__
#define __VKLLAMA_EMBEDDING_H__
#include "src/ops/op.h"
#include <memory>

class GPUDevice;
class Command;
class Embedding : public Op
{
public:
  Embedding (GPUDevice *dev, Command *, const uint32_t UNK);
  VkResult init () override;
  uint64_t time () override;
  VkResult operator() (VkTensor vocab, VkTensor indices, VkTensor &out);

private:
  std::unique_ptr<Pipeline> pipeline_;
  const uint32_t UNK_;
};
#endif
