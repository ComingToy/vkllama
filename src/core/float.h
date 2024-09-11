#ifndef __VKLLAMA_MATH_H__
#define __VKLLAMA_MATH_H__

#include <math.h>
#include <stdint.h>

typedef union
{
  struct
  {
    uint16_t frac : 10;
    uint16_t exp : 5;
    uint16_t sign : 1;
  } __attribute__ ((packed)) bits;

  uint16_t u16;
} __attribute__ ((packed)) __pack16_t;

typedef union
{
  struct
  {
    uint32_t frac : 23;
    uint32_t exp : 8;
    uint32_t sign : 1;
  } __attribute__ ((packed)) bits;
  uint32_t u32;
  float fp32;
} __attribute__ ((packed)) __pack32_t;

inline __pack16_t
__fp32_to_fp16 (float fp32)
{
  const float fp32_abs = fabs (fp32);
  __pack32_t pack32 = { .fp32 = fp32 };
  __pack16_t pack16 = { .u16 = 0 };

  if (pack32.bits.exp == 0 && pack32.bits.frac == 0)
    {
      pack16.bits.sign = pack32.bits.sign;
      pack16.bits.frac = 0;
      pack16.bits.exp = 0;
      return pack16;
    }

  // nan
  if (isnan (fp32))
    {
      pack16.bits.exp = 0x1f;
      pack16.bits.frac = 1;
      pack16.bits.sign = pack32.bits.sign;
      return pack16;
    }

  // inf
  if (isinf (fp32))
    {
      pack16.bits.exp = 0x1f;
      pack16.bits.frac = 0;
      pack16.bits.sign = pack32.bits.sign;
      return pack16;
    }

  // upper to fp16 max norm
  if (fp32_abs > 65504.0f)
    {
      pack16.bits.sign = pack32.bits.sign;
      pack16.bits.exp = 0x1e;
      pack16.bits.frac = 1023;
      return pack16;
    }

  // lower than min subnormalnorm
  if (fp32_abs < 5.96046448e-8f)
    {
      __pack16_t zero = { .u16 = 0 };
      return zero;
    }

  // lower than fp16 min norm: fp32 normalized to fp16 subnormal
  if (fp32_abs < 6.103515625e-5)
    {
      pack16.bits.sign = pack32.bits.sign;
      pack16.bits.exp = 0;
      // borrow from exp
      pack16.bits.frac = (pack32.bits.frac >> 14) + 512;
      return pack16;
    }

  // fp32 normalized to fp16 normalzied

  pack16.bits.sign = pack32.bits.sign;
  pack16.bits.exp = pack32.bits.exp - 127 + 15;
  pack16.bits.frac = pack32.bits.frac >> 13;
  return pack16;
}

inline float
__fp16_to_fp32 (uint16_t const value)
{
  __pack16_t pack16 = { .u16 = value };
  __pack32_t pack32 = { .u32 = 0 };

  if (pack16.bits.exp == 0 && pack16.bits.frac == 0)
    {
      return pack16.bits.sign == 0 ? .0f : -.0f;
    }

  // normalized case
  if (pack16.bits.exp != 0xff && pack16.bits.exp != 0)
    {
      pack32.bits.sign = pack16.bits.sign;
      pack32.bits.exp = pack16.bits.exp - 15 + 127;
      pack32.bits.frac = pack16.bits.frac << 13;
      return pack32.fp32;
    }

  // subnormal case
  // 5.96046448e-8f = 2**-14 * 1/1024.0
  if (pack16.bits.exp == 0 && pack16.bits.frac != 0)
    {
      const float alpha
          = pack16.bits.sign == 0 ? 5.96046448e-8f : -5.96046448e-8f;
      return pack16.bits.frac * alpha;
    }

  if (pack16.bits.exp == 0x1f && pack16.bits.frac == 0)
    {
      pack32.bits.sign = pack16.bits.sign;
      pack32.bits.exp = 0xff;
      pack32.bits.frac = 0;
      return pack32.fp32;
    }

  if (pack16.bits.exp == 0x1f && pack16.bits.frac != 0)
    {
      pack32.bits.sign = pack16.bits.sign;
      pack32.bits.exp = 0xff;
      pack32.bits.frac = 1;
      return pack32.fp32;
    }

  return pack32.fp32;
}

typedef __pack16_t __vkllama_fp16_t;
#endif
