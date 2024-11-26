#ifndef __VKLLAMA_SLICE_H__
#define __VKLLAMA_SLICE_H__

#include "src/ops/op.h"
#include <array>
#include <memory>
#include <vector>
namespace vkllama
{
class Slice : public Op
{
public:
  Slice (GPUDevice *gpu_, Command *command_, Tensor::DType dtype);
  absl::Status init () noexcept override;
  uint64_t time () noexcept override;
  absl::StatusOr<Tensor>
  operator() (Tensor in, std::array<uint32_t, 3> const &starts,
              std::array<uint32_t, 3> const &extents) noexcept;

private:
  Tensor::DType dtype_;
  std::unique_ptr<Pipeline> pipeline_;
  Tensor out_;
};
}
#endif
