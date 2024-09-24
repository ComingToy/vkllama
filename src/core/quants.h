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

struct DTypeProperty
{
  uint32_t items_per_block;
  uint32_t bytes_per_block;
};

extern DTypeProperty get_dtype_property (DType const dtype);
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
qint8_0_quantize_row (const T *src, int8_t *dst, const size_t n)
{
  const auto q8_0_property = get_dtype_property (Q8_0);
  const size_t block_counts = (n + q8_0_property.items_per_block - 1)
                              / q8_0_property.items_per_block;

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
      size_t start = b * q8_0_property.items_per_block;
      size_t end = start + q8_0_property.items_per_block;

      float max_abs_val = fabsf (load_fp (src, start));

      for (auto i = start; i < end; ++i)
        {
          if (i >= n)
            {
              break;
            }
          max_abs_val = std::max (fabsf (load_fp (src, i)), max_abs_val);
        }

      float scale = max_abs_val / 127.0f;
      float inverse_scale = scale > 0 ? 127.0f / max_abs_val : .0f;

      *((float *)write_dst) = scale;

      write_dst += 4;

      for (auto i = start; i < end; ++i)
        {
          if (i >= n)
            {
              *write_dst = int8_t (0);
            }
          else
            {
              auto v = roundf (load_fp (src, i) * inverse_scale);
              *write_dst = (int8_t)v;
            }
          ++write_dst;
        }
    }

  return absl::OkStatus ();
}

template <typename T>
absl::Status
qint8_0_dequantize_row (const int8_t *src, T *dst, const size_t n)
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

  auto const q8_0_property = get_dtype_property (Q8_0);
  const size_t block_counts = (n + q8_0_property.items_per_block - 1)
                              / q8_0_property.items_per_block;

  for (size_t b = 0; b < block_counts; ++b)
    {
      const int8_t *block = src + b * q8_0_property.bytes_per_block;

      float d = *reinterpret_cast<const float *> (block);

      block += sizeof (float);

      for (size_t i = 0; i < q8_0_property.items_per_block; ++i)
        {
          auto offset = b * q8_0_property.items_per_block + i;
          if (offset >= n)
            {
              break;
            }
          float v = block[i] * d;
          store_fp (dst, offset, v);
        }
    }

  return absl::OkStatus ();
}

template <typename T>
absl::Status
qint8_0_quantize (const T *src, int8_t *dst, const size_t h, const size_t w)
{
  const auto property = get_dtype_property (Q8_0);
  const size_t blocks
      = (w + property.items_per_block - 1) / property.items_per_block;

  const auto row_bytes = blocks * property.bytes_per_block;

  for (size_t i = 0; i < h; ++i)
    {
      auto *p = dst + i * row_bytes;
      auto s = qint8_0_quantize_row (src + i * w, p, w);
      if (!s.ok ())
        {
          return s;
        }
    }

  return absl::OkStatus ();
}

template <typename T>
absl::Status
qint8_0_dequantize (const int8_t *src, T *dst, const size_t h, const size_t w)
{
  const auto property = get_dtype_property (Q8_0);
  const size_t blocks
      = (w + property.items_per_block - 1) / property.items_per_block;

  const auto row_bytes = blocks * property.bytes_per_block;
  for (size_t i = 0; i < h; ++i)
    {
      auto *p = src + i * row_bytes;
      auto s = qint8_0_dequantize_row (p, dst + i * w, w);
      if (!s.ok ())
        {
          return s;
        }
    }

  return absl::OkStatus ();
}
}
#endif
