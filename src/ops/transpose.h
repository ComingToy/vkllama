#ifndef __VKLLAMA_TRANSPOSE_H__
#define __VKLLAMA_TRANSPOSE_H__
#include "src/core/pipeline.h"
#include "src/ops/op.h"
#include <memory>
#include <vector>

namespace vkllama
{
class Transpose : public Op
{
public:
  // type 0: axis = [1, 0, 2]
  Transpose (GPUDevice *gpu, Command *command, const int type,
             Tensor::DType const dtype = FP16);

  uint64_t time () noexcept override;
  absl::Status init () noexcept override;
  absl::StatusOr<Tensor> operator() (Tensor in) noexcept;

private:
  std::unique_ptr<Pipeline> pipeline_;
  const int dtype_;
  const int trans_type_;
  Tensor out_;
};
}
#endif
