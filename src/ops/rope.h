#ifndef __VKLLAMA_ROPE_H__
#define __VKLLAMA_ROPE_H__

#include "src/core/float.h"
#include "src/ops/op.h"
#include <memory>
#include <vector>

namespace vkllama
{
class Rope : public Op
{
public:
  Rope (GPUDevice *dev, Command *command, const int maxlen, const int dim,
        const Tensor::DType dtype = FP16);

  absl::StatusOr<Tensor> operator() (Tensor query,
                                     const size_t offset = 0) noexcept;
  absl::Status init () noexcept override;
  uint64_t time () noexcept override;

private:
  const int maxlen_;
  const int dim_;
  const Tensor::DType dtype_;

  std::unique_ptr<Pipeline> pipeline_k_;
  std::unique_ptr<Pipeline> pipeline_q_;
};

}

#endif
