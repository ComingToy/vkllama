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
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "src/core/float.h"
#include "src/core/quants.h"
#include <errno.h>
#include <map>

ABSL_FLAG (std::string, in, "", "path to input gguf file");
ABSL_FLAG (std::string, out, "", "path to output gguf file");
ABSL_FLAG (std::string, type, "", "quantize type. q8_0");

static absl::StatusOr<std::tuple<size_t, size_t, size_t> >
get_tensor_shape (const gguf_tensor *tensor)
{
  if (tensor->ndim > 3)
    {
      return absl::InvalidArgumentError (
          absl::StrFormat ("%d dims is unsupported.", (int)tensor->ndim));
    }

  // c, h , w
  std::tuple<size_t, size_t, size_t> dims;

  switch (tensor->ndim)
    {
    case 3:
      std::get<0> (dims) = tensor->dim[2];
    case 2:
      std::get<1> (dims) = tensor->dim[1];
    case 1:
      std::get<2> (dims) = tensor->dim[0];
    default:
      break;
    }

  return absl::OkStatus ();
}

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
  const auto q8_0_property = vkllama::get_dtype_property (vkllama::Q8_0);

  std::map<std::string, size_t> offsets;

  for (auto &[name, tensor] : tensors)
    {
      tensor_offset
          += gguf_get_alignment_padding (gguf->alignment, tensor_offset);

      auto type = tensor.type;
      auto tensor_size = tensor.bsize;

      auto skip = name.find ("_norm.weight") != std::string::npos;

      auto shape = get_tensor_shape (&tensor);
      if (!shape.ok ())
        {
          return shape.status ();
        }

      auto [c, h, w] = *shape;

      if ((tensor.type == GGUF_TYPE_F16 || tensor.type == GGUF_TYPE_F32)
          && !skip)
        {
          type = GGUF_TYPE_Q8_0;

          w = (w + q8_0_property.items_per_block - 1)
              / q8_0_property.items_per_block * q8_0_property.items_per_block;
          auto n = w * c * h;

          auto block_counts = n / q8_0_property.items_per_block;

          tensor_size = block_counts * q8_0_property.bytes_per_block;
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

      auto shape = get_tensor_shape (&tensor);
      if (!shape.ok ())
        {
          return shape.status ();
        }

      auto [c, h, w] = *shape;

      auto aligned_w = (w + q8_0_property.items_per_block - 1)
                       / q8_0_property.items_per_block
                       * q8_0_property.items_per_block;

      auto block_counts = (c * h * aligned_w) / q8_0_property.items_per_block;

      std::vector<int8_t> quantized_weights (block_counts
                                             * q8_0_property.bytes_per_block);

      auto status = absl::OkStatus ();

      if (tensor.type == GGUF_TYPE_F32)
        {

          status = vkllama::qint8_0_quantize (
              (const float *)tensor.weights_data, quantized_weights.data (),
              c * h, w);
        }
      else
        {
          status = vkllama::qint8_0_quantize (
              (const __vkllama_fp16_t *)tensor.weights_data,
              quantized_weights.data (), c * h, w);
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

  if (type == "q8_0")
    {
      gguf_ctx *gguf_quantized = gguf_create (out.c_str (), GGUF_OVERWRITE);
      if (!gguf_quantized)
        {
          fprintf (stderr, "create output gguf file error: %s",
                   strerror (errno));
          return -1;
        }

      auto s = qint8_0_quantize (gguf_quantized, meta, kv_lens, tensors, 1);
      if (!s.ok ())
        {
          std::cerr << "q8_0_quantize failed: " << s << std::endl;
          return -1;
        }
    }

  return 0;
}
