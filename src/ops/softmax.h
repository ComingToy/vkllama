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
           VkTensor::DType const dtype = VkTensor::FP32);
  VkResult init () noexcept override;
  uint64_t time () noexcept override;
  VkResult operator() (VkTensor a, VkTensor &b, size_t offset = 0) noexcept;

private:
  std::unique_ptr<Reduce> reduce_;
  std::unique_ptr<Pipeline> softmax0_;
  std::unique_ptr<Pipeline> softmax1_;
  std::unique_ptr<Pipeline> softmax2_;

  VkTensor bias_;
  VkTensor m_;
  VkTensor exps_;
  VkTensor n_;
  VkTensor sum_;
  bool seq_mask_;
  VkTensor::DType dtype_;
};

}
#endif
