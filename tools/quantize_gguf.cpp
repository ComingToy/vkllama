// clang-format off
#include <algorithm>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <type_traits>
#include <vector>
extern "C"
{
#include "gguflib.h"
}
// clang-format on
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "src/core/float.h"
#include <map>

ABSL_FLAG (std::string, in, "", "path to input gguf file");
ABSL_FLAG (std::string, out, "", "path to output gguf file");
ABSL_FLAG (std::string, type, "", "quantize type. int8_0|int8_1");

static absl::Status
read_gguf (gguf_ctx *gguf, std::map<std::string, gguf_key> &meta,
           std::map<std::string, gguf_tensor> &tensors)
{
  gguf_key key;
  while (::gguf_get_key (gguf, &key))
    {
      printf ("%.*s: [%s] ", (int)key.namelen, key.name,
              gguf_get_value_type_name (key.type));
      gguf_print_value (gguf, key.type, key.val, 0);
      printf ("\n");

      std::string name (key.name, key.namelen);
      meta[name] = key;
    }

  gguf_tensor tensor;
  while (gguf_get_tensor (gguf, &tensor))
    {
      std::string name (tensor.name, tensor.namelen);
      tensors[name] = tensor;
      if (tensor.type != GGUF_TYPE_F16 && tensor.type != GGUF_TYPE_F32)
        {
          return absl::InvalidArgumentError (absl::StrFormat (
              "model weights must be float point type. dtype of tensor %s is "
              "%s\n",
              name.c_str (), gguf_get_tensor_type_name (tensor.type)));
        }
    }

  return absl::OkStatus ();
}

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
                        const size_t block_size)
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

      __vkllama_fp16_t scale16 = __fp32_to_fp16 (scale);
      *((uint16_t *)write_dst) = scale16.u16;
      write_dst += 2;

      for (auto i = start; i < end; ++i)
        {
          write_dst[i] = (int8_t)roundf (load_fp (src, i) * inverse_scale);
        }
    }

  return absl::OkStatus ();
}

absl::Status
qint8_0_quantize (gguf_ctx *gguf, std::map<std::string, gguf_key> &meta,
                  std::map<std::string, gguf_tensor> &tensors)
{
  constexpr size_t items_per_block = 32;
  for (auto &[name, tensor] : tensors)
    {
      if (tensor.type != GGUF_TYPE_F16 && tensor.type != GGUF_TYPE_F32)
        {
          continue;
        }

      auto block_counts
          = (tensor.num_weights + items_per_block - 1) / items_per_block;
      std::vector<int8_t> quantized_weights (block_counts
                                             * (items_per_block + 2));

      auto status = absl::OkStatus ();

      if (tensor.type == GGUF_TYPE_F32)
        {
          status = qint8_0_quantize_block (
              (const float *)tensor.weights_data, quantized_weights.data (),
              tensor.num_weights, items_per_block);
        }
      else
        {
          status = qint8_0_quantize_block (
              (const __vkllama_fp16_t *)tensor.weights_data,
              quantized_weights.data (), tensor.num_weights, items_per_block);
        }

      if (!status.ok ())
        return status;
      ;
    }

  return absl::OkStatus ();
}

int
main (int argc, char *argv[])
{
  absl::ParseCommandLine (argc, argv);

  auto in = absl::GetFlag (FLAGS_in);
  auto out = absl::GetFlag (FLAGS_out);
  auto type = absl::GetFlag (FLAGS_type);

  if (in.empty () || out.empty () || type.empty ())
    {
      fprintf (stdout,
               "usage: %s --in <path to input gguf> --out <path to output "
               "gguf> --type <quantization type>\n",
               argv[0]);
      return -1;
    }

  fprintf (stdout, "input gguf: %s\noutput gguf: %s\nquantization type: %s\n",
           in.c_str (), out.c_str (), type.c_str ());

  auto gguf = gguf_open (in.c_str ());
  if (!gguf)
    {
      fprintf (stderr, "open model file %s failed.\n", in.c_str ());
      return -1;
    }

  std::map<std::string, gguf_key> meta;
  std::map<std::string, gguf_tensor> tensors;
  if (auto ret = read_gguf (gguf, meta, tensors); !ret.ok ())
    {
      std::cerr << "read gguf failed: " << ret << std::endl;
      return -1;
    }

  if (type == "int8_0")
    {
      auto s = qint8_0_quantize (gguf, meta, tensors);
      if (!s.ok ())
        {
          std::cerr << "qint8_0_quantize failed: " << s << std::endl;
          return -1;
        }
    }

  return 0;
}
