#ifndef __VKLLAMA_RMSNORM_H__
#define __VKLLAMA_RMSNORM_H__
#include "op.h"
#include "src/core/tensor.h"
#include <memory>

class RMSNorm : public Op
{
public:
  RMSNorm (GPUDevice *dev, Command *command);
  VkResult operator() (VkTensor a, VkTensor b, VkTensor &c) noexcept;
  VkResult init () noexcept override;
  uint64_t time () noexcept override;

private:
  std::unique_ptr<Pipeline> pipeline_;
};
#endif
