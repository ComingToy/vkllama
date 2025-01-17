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
               Tensor::DType dtype = FP16);
  absl::StatusOr<Tensor> operator() (Tensor x, Tensor y) noexcept;
  absl::StatusOr<Tensor> operator() (Tensor x, float y) noexcept;
  absl::Status init () noexcept override;
  uint64_t time () noexcept override;

private:
  const int type_;
  std::unique_ptr<Pipeline> pipeline0_;
  std::unique_ptr<Pipeline> pipeline1_;
  Tensor::DType dtype_;
};

}

#endif
