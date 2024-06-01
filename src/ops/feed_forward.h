#ifndef __VKLLAMA_FEED_FORWARD__
#define __VKLLAMA_FEED_FORWARD__

#include "src/core/tensor.h"
#include "src/ops/elementwise.h"
#include "src/ops/mat_mul.h"
#include "src/ops/op.h"
#include <memory>

class GPUDevice;
class Command;
class FeedForward : public Op
{
public:
  FeedForward (GPUDevice *, Command *, VkTensor, VkTensor, VkTensor,
               const bool transposed_weight = false,
               const VkTensor::DType dtype = VkTensor::FP32);
  VkResult operator() (VkTensor X, VkTensor &output) noexcept;
  VkResult init () noexcept override;
  uint64_t time () noexcept override;

private:
  VkTensor t0_;
  VkTensor t1_;
  VkTensor t2_;

  VkTensor w1_;
  VkTensor w2_;
  VkTensor w3_;
  std::unique_ptr<MatMul> up_op_;
  std::unique_ptr<MatMul> down_op_;
  std::unique_ptr<MatMul> gate_op_;
  std::unique_ptr<ElementWise> elemwise_op_;
  VkTensor::DType dtype_;

  bool transposed_weight_;
};

#endif
