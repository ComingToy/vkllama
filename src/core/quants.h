#ifndef __VKLLAMA_QUANTS_H__
#define __VKLLAMA_QUANTS_H__

#include "absl/status/status.h"
#include "src/core/float.h"
#include <algorithm>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <type_traits>
#include <vector>

namespace vkllama
{

typedef enum : int
{
  FP32 = 0,
  FP16,
  UINT32,
  INT8,
  Q8_0, // block-wise quantize
} DType;

struct vkllama_dtype_property
{
  const char *name;
  uint32_t items_per_block;
  uint32_t bytes_per_block;
};

/**
 * @brief quantize float weights to int8.
 *
 * @param src fp32 weight data
 * @param dst q8_0 format block. mem layout [<fp16 scale>, int8 weights...]
 * @param n num of fp32 weights
 * @param block_size
 *
 * @return
 */
template <typename T>
absl::Status
qint8_0_quantize_block (const T *src, int8_t *dst, const size_t n,
                        const size_t block_size, const int d_type = 0)
{
  const size_t block_counts = (n + block_size - 1) / block_size;

  int8_t *write_dst = dst;

  auto load_fp = [] (const T *src, size_t i) {
    if constexpr (std::is_same<typename std::remove_const<T>::type,
                               __vkllama_fp16_t>::value)
      {
        return __fp16_to_fp32 (src[i].u16);
      }
    else
      {
        return src[i];
      }
  };

  for (size_t b = 0; b < block_counts; ++b)
    {
      size_t start = b * block_size;
      size_t end = std::min (start + block_size, n);

      float max_abs_val = fabsf (load_fp (src, start));

      for (auto i = start; i < end; ++i)
        {
          max_abs_val = std::max (fabsf (load_fp (src, i)), max_abs_val);
        }

      float scale = max_abs_val / 127.0f;
      float inverse_scale = scale > 0 ? 127.0f / max_abs_val : .0f;

      if (d_type == 0)
        {
          __vkllama_fp16_t scale16 = __fp32_to_fp16 (scale);
          *((uint16_t *)write_dst) = scale16.u16;
        }
      else if (d_type == 1)
        {
          *((float *)write_dst) = scale;
        }

      write_dst += 4;

      for (auto i = start; i < end; ++i)
        {
          auto v = roundf (load_fp (src, i) * inverse_scale);
          *write_dst = (int8_t)v;
          ++write_dst;
        }
    }

  return absl::OkStatus ();
}

template <typename T>
absl::Status
qint8_0_dequantize_block (const int8_t *src, T *dst, const size_t n,
                          const size_t block_size, int d_type)
{
  auto store_fp = [] (T *buf, size_t i, const float val) {
    if constexpr (std::is_same<typename std::remove_const<T>::type,
                               __vkllama_fp16_t>::value)
      {
        buf[i] = __fp32_to_fp16 (val);
      }
    else
      {
        buf[i] = val;
      }
  };

  const size_t d_size = d_type ? 4 : 2;
  const size_t block_counts = (n + block_size - 1) / block_size;

  for (size_t b = 0; b < block_counts; ++b)
    {
      const int8_t *block = src + b * (block_size + d_size);

      float d = .0f;
      if (d_type == 0)
        {
          const __vkllama_fp16_t *p
              = reinterpret_cast<const __vkllama_fp16_t *> (block);
          d = __fp16_to_fp32 (p->u16);
        }
      else
        {
          d = *reinterpret_cast<const float *> (block);
        }

      block += d_size;

      for (size_t i = 0; i < block_size; ++i)
        {
          if (b * block_size + i >= n)
            break;

          float v = block[i] * d;
          store_fp (dst, b * block_size + i, v);
        }
    }

  return absl::OkStatus ();
}
}
#endif
