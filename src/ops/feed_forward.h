#ifndef __VKLLAMA_FEED_FORWARD__
#define __VKLLAMA_FEED_FORWARD__

#include "src/core/tensor.h"
#include "src/ops/op.h"
#include <memory>

class GPUDevice;
class Command;
class FeedForward : public Op
{
public:
  FeedForward (GPUDevice *, Command *, VkTensor, VkTensor, VkTensor);
  VkResult operator() (VkTensor X, VkTensor &output) noexcept;
  VkResult init () noexcept override;
  uint64_t time () noexcept override;

private:
  VkTensor w1_;
  VkTensor w2_;
  VkTensor w3_;

  VkTensor t0_;
  VkTensor t1_;
  VkTensor t2_;
  std::unique_ptr<Pipeline> pipeline0_;
  std::unique_ptr<Pipeline> pipeline1_;
  std::unique_ptr<Pipeline> pipeline2_;
  std::unique_ptr<Pipeline> pipeline3_;
};

#endif
