#include "op.h"
#include "src/core/pipeline.h"

Op::Op(GPUDevice* dev, Command* command)
  : dev_(dev)
  , command_(command)
{
}

VkResult
Op::init()
{
    return pipeline_->init();
}

uint64_t
Op::time()
{
    return pipeline_->time();
}
