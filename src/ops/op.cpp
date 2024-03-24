#include "op.h"
#include "src/core/pipeline.h"

Op::Op(GPUDevice* dev, Command* command)
  : dev_(dev)
  , command_(command)
{
}

