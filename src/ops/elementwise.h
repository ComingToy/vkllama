#ifndef __VKLLAMA_ELEMENTWISE_H__
#define __VKLLAMA_ELEMENTWISE_H__
#include "src/core/float.h"
#include "src/ops/op.h"
#include <memory>

namespace vkllama
{
class Pipeline;
class GPUDevice;
class Command;

class ElementWise : public Op
{
public:
  ElementWise (GPUDevice *dev, Command *command_, const int type,
               VkTensor::DType dtype = VkTensor::FP32);
  VkResult operator() (VkTensor x, VkTensor y, VkTensor &out) noexcept;
  VkResult operator() (VkTensor x, float y, VkTensor &out) noexcept;
  VkResult init () noexcept override;
  uint64_t time () noexcept override;

private:
  const int type_;
  std::unique_ptr<Pipeline> pipeline0_;
  std::unique_ptr<Pipeline> pipeline1_;
  VkTensor::DType dtype_;
};

}

#endif
