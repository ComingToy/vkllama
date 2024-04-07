#ifndef __VKLLAMA_CONCAT_H__
#define __VKLLAMA_CONCAT_H__
#include "src/core/pipeline.h"
#include "src/ops/op.h"
#include <memory>
#include <vector>

class Concat : public Op
{
public:
  Concat (GPUDevice *gpu, Command *command, const int num);
  VkResult init () override;
  VkResult operator() (std::vector<VkTensor> const &inputs, VkTensor &output);
  uint64_t time () override;

private:
  const int num_;
  std::vector<std::unique_ptr<Pipeline> > pipelines_;
};
#endif
