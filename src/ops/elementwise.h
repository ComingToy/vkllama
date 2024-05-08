#ifndef __VKLLAMA_ELEMENTWISE_H__
#define __VKLLAMA_ELEMENTWISE_H__
#include "src/ops/op.h"
#include <memory>
class Pipeline;
class GPUDevice;
class Command;
class ElementWise : public Op
{
public:
  ElementWise (GPUDevice *dev, Command *command_, const int type);
  VkResult operator() (VkTensor x, VkTensor y, VkTensor &out) noexcept;
  VkResult operator() (VkTensor x, float y, VkTensor &out) noexcept;
  VkResult init () noexcept override;
  uint64_t time () noexcept override;

private:
  const int type_;
  std::unique_ptr<Pipeline> pipeline0_;
  std::unique_ptr<Pipeline> pipeline1_;
};

#endif
