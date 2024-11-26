#ifndef __VKLLAMA_CAST_H__
#define __VKLLAMA_CAST_H__
#include "src/core/pipeline.h"
#include "src/ops/op.h"
#include <memory>

namespace vkllama
{
class Pipeline;
class GPUDevice;
class Command;
class Cast : public Op
{
public:
  Cast (GPUDevice *dev, Command *command, const Tensor::DType from,
        const Tensor::DType to);

  absl::StatusOr<Tensor> operator() (Tensor from) noexcept;
  absl::Status init () noexcept override;
  uint64_t time () noexcept override;

private:
  Tensor::DType from_;
  Tensor::DType to_;
  Tensor out_;
  std::unique_ptr<Pipeline> pipeline_;
};

}

#endif
