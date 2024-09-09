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
        const VkTensor::DType dtype = VkTensor::FP32);
  VkResult operator() (VkTensor query, VkTensor key, VkTensor &out_query,
                       VkTensor &out_key, const size_t offset = 0) noexcept;
  VkResult init () noexcept override;
  uint64_t time () noexcept override;

private:
  const int maxlen_;
  const int dim_;
  const VkTensor::DType dtype_;

  std::unique_ptr<Pipeline> pipeline_k_;
  std::unique_ptr<Pipeline> pipeline_q_;
};

}

#endif
