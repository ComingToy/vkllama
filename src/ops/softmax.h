#ifndef __VKLLAMA_SOFTMAX_H__
#define __VKLLAMA_SOFTMAX_H__
#include "src/ops/op.h"
#include "src/ops/reduce.h"
#include <memory>

namespace vkllama
{
class GPUDevice;
class Command;
class Softmax : public Op
{
public:
  Softmax (GPUDevice *dev, Command *command, bool seq_mask = false,
           const float temp = 1.0,
           VkTensor::DType const dtype = VkTensor::FP32);
  VkResult init () noexcept override;
  uint64_t time () noexcept override;
  VkResult operator() (VkTensor a, VkTensor &b, size_t offset = 0) noexcept;

private:
  std::unique_ptr<Pipeline> softmax0_;

  bool seq_mask_;
  VkTensor::DType dtype_;

  float temp_;
};

}
#endif
