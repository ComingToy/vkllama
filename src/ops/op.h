#ifndef __VKLLAMA_OP_H__
#define __VKLLAMA_OP_H__

#include "src/core/pipeline.h"
#include "src/core/tensor.h"

class GPUDevice;
class Command;

class Op
{
  public:
    Op(GPUDevice* dev, Command* command);
    virtual VkResult init() = 0;
    virtual uint64_t time() = 0;

  protected:
    GPUDevice* dev_;
    Command* command_;
};

#endif
