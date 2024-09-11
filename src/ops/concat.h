#ifndef __VKLLAMA_CONCAT_H__
#define __VKLLAMA_CONCAT_H__
#include "src/core/pipeline.h"
#include "src/ops/op.h"
#include <memory>
#include <vector>

namespace vkllama
{
class Concat : public Op
{
public:
  Concat (GPUDevice *gpu, Command *command, const int num, const int axis,
          Tensor::DType const dtype = Tensor::FP32);
  absl::Status init () noexcept override;
  absl::StatusOr<Tensor>
  operator() (std::vector<Tensor> const &inputs) noexcept;
  uint64_t time () noexcept override;

private:
  const int num_;
  std::vector<std::unique_ptr<Pipeline> > pipelines_;
  int axis_;
  Tensor::DType dtype_;
};

}
#endif
