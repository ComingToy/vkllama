// clang-format off
#include <algorithm>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
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
#include <errno.h>
#include <map>

ABSL_FLAG (std::string, in, "", "path to input gguf file");
ABSL_FLAG (std::string, out, "", "path to output gguf file");
ABSL_FLAG (std::string, type, "", "quantize type. int8_0|int8_1|int8_2");

static absl::Status
read_gguf (gguf_ctx *gguf, std::map<std::string, gguf_key> &meta,
           std::map<std::string, size_t> &kv_lens,
           std::map<std::string, gguf_tensor> &tensors)
{
  gguf_key key;
  while (::gguf_get_key (gguf, &key))
    {
      printf ("%.*s: [%s] ", (int)key.namelen, key.name,
              gguf_get_value_type_name (key.type));

      auto value_start = gguf->off;
      gguf_print_value (gguf, key.type, key.val, 0);
      auto value_len = gguf->off - value_start;

      printf ("\n");

      std::string name (key.name, key.namelen);
      meta[name] = key;
      kv_lens[name] = value_len;
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

absl::Status
qint8_0_quantize (gguf_ctx *gguf, std::map<std::string, gguf_key> &meta,
                  std::map<std::string, size_t> &kv_lens,
                  std::map<std::string, gguf_tensor> &tensors, int d_type = 0)
{

  for (auto const &[name, key] : meta)
    {
      if (!kv_lens.count (name))
        {
          return absl::InternalError (
              absl::StrFormat ("value length of %s is missed", name));
        }

      gguf_append_kv (gguf, key.name, key.namelen, key.type, key.val,
                      kv_lens[name]);
    }

  size_t tensor_offset = 0;
  constexpr size_t items_per_block = 32;
  std::map<std::string, size_t> offsets;

  for (auto &[name, tensor] : tensors)
    {
      tensor_offset
          += gguf_get_alignment_padding (gguf->alignment, tensor_offset);

      auto type = tensor.type;
      auto tensor_size = tensor.bsize;

      auto skip = name.find ("_norm.weight") != std::string::npos;

      if ((tensor.type == GGUF_TYPE_F16 || tensor.type == GGUF_TYPE_F32)
          && !skip)
        {
          type = GGUF_TYPE_Q8_0;

          auto block_counts
              = (tensor.num_weights + items_per_block - 1) / items_per_block;
          tensor_size = block_counts * (items_per_block + 4);
        }

      auto ret = gguf_append_tensor_info (gguf, tensor.name, tensor.namelen,
                                          tensor.ndim, tensor.dim, type,
                                          tensor_offset);
      if (ret == 0)
        {
          return absl::InternalError (
              absl::StrFormat ("failed to append %s tensor info", name));
        }

      fprintf (stderr,
               "append tensor info, name = %s, ndim = %d, dims = [%lu, %lu, "
               "%lu, %lu], type = %s, offset = %lu\n",
               name.c_str (), (int)tensor.ndim, tensor.dim[0], tensor.dim[1],
               tensor.dim[2], tensor.dim[3], gguf_get_tensor_type_name (type),
               tensor_offset);

      tensor_offset += tensor_size;
    }

  for (auto &[name, tensor] : tensors)
    {
      if (tensor.type != GGUF_TYPE_F16 && tensor.type != GGUF_TYPE_F32)
        {
          auto ret = gguf_append_tensor_data (gguf, tensor.weights_data,
                                              tensor.num_weights);
          if (!ret)
            {
              return absl::InternalError (absl::StrFormat (
                  "failed to append %s quantized tensor data", name));
            }

          continue;
        }

      auto block_counts
          = (tensor.num_weights + items_per_block - 1) / items_per_block;
      std::vector<int8_t> quantized_weights (block_counts
                                             * (items_per_block + 4));

      auto status = absl::OkStatus ();

      if (tensor.type == GGUF_TYPE_F32)
        {
          status = qint8_0_quantize_block (
              (const float *)tensor.weights_data, quantized_weights.data (),
              tensor.num_weights, items_per_block, d_type);
        }
      else
        {
          status = qint8_0_quantize_block (
              (const __vkllama_fp16_t *)tensor.weights_data,
              quantized_weights.data (), tensor.num_weights, items_per_block,
              d_type);
        }

      if (!status.ok ())
        return status;

      auto ret = gguf_append_tensor_data (
          gguf, (void *)quantized_weights.data (), quantized_weights.size ());
      if (!ret)
        {
          return absl::InternalError (absl::StrFormat (
              "failed to append %s quantized tensor data", name));
        }
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
  std::map<std::string, size_t> kv_lens;

  std::map<std::string, gguf_tensor> tensors;
  if (auto ret = read_gguf (gguf, meta, kv_lens, tensors); !ret.ok ())
    {
      std::cerr << "read gguf failed: " << ret << std::endl;
      return -1;
    }

  if (type == "int8_0")
    {
      gguf_ctx *gguf_quantized = gguf_create (out.c_str (), GGUF_OVERWRITE);
      if (!gguf_quantized)
        {
          fprintf (stderr, "create output gguf file error: %s",
                   strerror (errno));
          return -1;
        }

      auto s = qint8_0_quantize (gguf_quantized, meta, kv_lens, tensors);
      if (!s.ok ())
        {
          std::cerr << "qint8_0_quantize failed: " << s << std::endl;
          return -1;
        }
    }

  return 0;
}
