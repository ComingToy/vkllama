#ifndef __VKLLAMA_RMSNORM_H__
#define __VKLLAMA_RMSNORM_H__
#include "op.h"
#include "src/core/tensor.h"
#include <memory>

namespace vkllama
{
class RMSNorm : public Op
{
public:
  RMSNorm (GPUDevice *dev, Command *command, Tensor weight,
           const float eps = 1e-3, const Tensor::DType dtype = FP16);
  absl::StatusOr<Tensor> operator() (Tensor a) noexcept;
  absl::Status init () noexcept override;
  uint64_t time () noexcept override;

private:
  std::unique_ptr<Pipeline> pipeline_;
  Tensor weight_;
  Tensor::DType dtype_;
  Tensor out_;
};

}
#endif
