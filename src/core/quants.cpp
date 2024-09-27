#include "src/core/quants.h"
#include <map>

namespace vkllama
{
DTypeProperty
get_dtype_property (const DType dtype)
{
  static std::map<DType, DTypeProperty> properties = {
    { FP32, { 1, sizeof (float) } },
    { FP16, { 1, sizeof (__vkllama_fp16_t) } },
    { UINT32, { 1, sizeof (uint32_t) } },
    { INT8, { 1, sizeof (int8_t) } },
    { Q8_0, { 32, 36 } },
  };

  return properties[dtype];
}
}
