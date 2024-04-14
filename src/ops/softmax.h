#ifndef __VKLLAMA_SOFTMAX_H__
#define __VKLLAMA_SOFTMAX_H__
#include "src/ops/op.h"
#include "src/ops/reduce.h"
#include <memory>

class GPUDevice;
class Command;
class Softmax : public Op
{
public:
  Softmax (GPUDevice *dev, Command *command);
  VkResult init () override;
  uint64_t time () override;
  VkResult operator() (VkTensor a, VkTensor &b);

private:
  std::unique_ptr<Reduce> reduce_;
  std::unique_ptr<Pipeline> softmax0_;
  std::unique_ptr<Pipeline> softmax1_;

  VkTensor bias_;
  VkTensor m_;
  VkTensor exps_;
  VkTensor n_;
  VkTensor sum_;
  VkTensor out_;
};
#endif
