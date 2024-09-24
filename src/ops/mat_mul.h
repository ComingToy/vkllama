#ifndef __VKLLAMA_MATMUL_H__
#define __VKLLAMA_MATMUL_H__

#include "src/ops/op.h"
#include <memory>

namespace vkllama
{
class MatMul : public Op
{
public:
  MatMul (GPUDevice *dev, Command *command, Tensor weight,
          const float scale = 1.0f, const float bias = .0f, const int act = 0,
          const int broadcast_type = 0, const bool transpose_b = false,
          const Tensor::DType a_dtype = FP16,
          const Tensor::DType b_dtype = FP16);

  MatMul (GPUDevice *dev, Command *command, const float scale = 1.0f,
          const float bias = .0f, const int act = 0,
          const int broadcast_type = 0, const bool transpose_b = false,
          const Tensor::DType a_dtype = FP16,
          const Tensor::DType b_dtype = FP16);

  absl::StatusOr<Tensor> operator() (Tensor c) noexcept;
  absl::StatusOr<Tensor> operator() (Tensor a, Tensor b) noexcept;
  absl::Status init () noexcept override;
  uint64_t time () noexcept override;

private:
  std::unique_ptr<Pipeline> pipeline_;
  Tensor weight_;
  const int broadcast_type_;
  const int act_;
  const bool transpose_b_;
  Tensor::DType a_dtype_;
  Tensor::DType b_dtype_;
  const float scale_;
  const float bias_;
  Pipeline::ShaderInfo shader_info_;
};
}

#endif
