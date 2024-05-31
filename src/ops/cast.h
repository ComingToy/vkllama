#ifndef __VKLLAMA_CAST_H__
#define __VKLLAMA_CAST_H__
#include "src/core/pipeline.h"
#include "src/ops/op.h"
#include <memory>

class Pipeline;
class GPUDevice;
class Command;
class Cast : public Op
{
public:
  Cast (GPUDevice *dev, Command *command, const VkTensor::DType from,
        const VkTensor::DType to);

  VkResult operator() (VkTensor from, VkTensor &to) noexcept;
  VkResult init () noexcept override;
  uint64_t time () noexcept override;

private:
  VkTensor::DType from_;
  VkTensor::DType to_;
  std::unique_ptr<Pipeline> pipeline_;
};

#endif
