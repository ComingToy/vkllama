#ifndef __VKLLAMA_COMMON_H__
#define __VKLLAMA_COMMON_H__
#include <stdint.h>
namespace vkllama
{
#define VKLLAMA_STATUS_OK(__expr)                                             \
  if (auto __s = (__expr); !__s.ok ())                                        \
    {                                                                         \
      return __s;                                                             \
    }

#ifndef __VKLLAMA_LOG_COST
#define __VKLLAMA_LOG_COST 0
#endif
typedef enum : int
{
  FP32 = 0,
  FP16,
  UINT32,
  INT8,
  Q8_0, // block-wise quantize
} DType;

struct ShapeConstant
{
  uint32_t c, h, w, cs, hs, ws;
} __attribute__ ((packed));
}

#endif
