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
             VkTensor::DType const dtype = VkTensor::FP32);

  uint64_t time () noexcept override;
  VkResult init () noexcept override;
  VkResult operator() (VkTensor in, VkTensor &out) noexcept;

private:
  std::unique_ptr<Pipeline> pipeline_;
  const int dtype_;
  const int trans_type_;
};
}
#endif
