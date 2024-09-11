#ifndef __VKLLAMA_FEED_FORWARD__
#define __VKLLAMA_FEED_FORWARD__

#include "src/core/tensor.h"
#include "src/ops/elementwise.h"
#include "src/ops/mat_mul.h"
#include "src/ops/op.h"
#include <memory>

namespace vkllama
{
class GPUDevice;
class Command;
class FeedForward : public Op
{
public:
  FeedForward (GPUDevice *, Command *, Tensor, Tensor, Tensor,
               const bool transposed_weight = false,
               const Tensor::DType dtype = Tensor::FP16);
  absl::StatusOr<Tensor> operator() (Tensor X) noexcept;
  absl::Status init () noexcept override;
  uint64_t time () noexcept override;

private:
  Tensor t0_;
  Tensor t1_;
  Tensor t2_;

  Tensor w1_;
  Tensor w2_;
  Tensor w3_;
  std::unique_ptr<MatMul> up_op_;
  std::unique_ptr<MatMul> down_op_;
  std::unique_ptr<MatMul> gate_op_;
  std::unique_ptr<ElementWise> elemwise_op_;
  Tensor::DType dtype_;

  bool transposed_weight_;
};

}

#endif
