#ifndef __VKLLAMA_OP_H__
#define __VKLLAMA_OP_H__

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/core/pipeline.h"
#include "src/core/tensor.h"

namespace vkllama
{
class GPUDevice;
class Command;

class Op
{
public:
  Op (GPUDevice *dev, Command *command) noexcept;
  virtual absl::Status init () noexcept = 0;
  virtual uint64_t time () noexcept = 0;
  virtual ~Op (){};

protected:
  GPUDevice *dev_;
  Command *command_;
};

}

#endif
