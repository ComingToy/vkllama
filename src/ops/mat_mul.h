#ifndef __VKLLAMA_MATMUL_H__
#define __VKLLAMA_MATMUL_H__

#include "src/ops/op.h"
#include <memory>

class MatMul: public Op
{
  public:
    MatMul(GPUDevice* dev, Command* command);
    VkResult operator()(VkTensor a, VkTensor b, VkTensor& c);
};

#endif
