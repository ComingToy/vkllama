// clang-format off
#include <stddef.h>
#include <stdio.h>
extern "C"
{
#include "gguflib.h"
}
// clang-format on
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
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

absl::Status
int8_0_quantize (gguf_ctx *gguf, std::map<std::string, gguf_key> &meta,
                 std::map<std::string, gguf_tensor> &tensors)
{

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
    }

  return 0;
}
