#include "op.h"
#include "src/core/pipeline.h"

namespace vkllama
{
Op::Op (GPUDevice *dev, Command *command) noexcept : dev_ (dev),
                                                     command_ (command)
{
}
}

