#ifndef __VKLLAMA_RMSNORM_V2_H__
#define __VKLLAMA_RMSNORM_V2_H__
#include "op.h"
#include "src/core/tensor.h"
#include <memory>

namespace vkllama
{
class RMSNormV2 : public Op
{
public:
  RMSNormV2 (GPUDevice *dev, Command *command, Tensor weight,
             const float eps = 1e-3,
             const Tensor::DType dtype = Tensor::FP32);
  VkResult operator() (Tensor a, Tensor &c) noexcept;
  VkResult init () noexcept override;
  uint64_t time () noexcept override;

private:
  std::unique_ptr<Pipeline> pipeline0_;
  std::unique_ptr<Pipeline> pipeline1_;
  std::unique_ptr<Pipeline> pipeline2_;
  Tensor weight_;
  Tensor::DType dtype_;
  Tensor stage0_out0_;
  Tensor stage0_out1_;
  Tensor stage1_out0_;
};

}
#endif
