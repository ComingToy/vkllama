#include "models/gguf/gguf.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <vector>

int
main (const int argc, const char *argv[])
{
  if (argc != 2)
    {
      fprintf (stderr, "usage %s <path to gguf>\n", argv[0]);
      return -1;
    }

#if 1
  GGUF gguf (argv[1]);
  if (gguf.init () != 0)
    {
      fprintf (stderr, "init gguf failed.\n");
      return -1;
    }

  /*
magic: 0x46554747
version: 2
tensor_count: 0
metadata_kv_count: 18
key: tokenizer.ggml.unknown_token_id, type: GGUF_METADATA_VALUE_TYPE_UINT32
key: tokenizer.ggml.eos_token_id, type: GGUF_METADATA_VALUE_TYPE_UINT32
key: tokenizer.ggml.merges, type: GGUF_METADATA_VALUE_TYPE_ARRAY
key: general.architecture, type: GGUF_METADATA_VALUE_TYPE_STRING
key: llama.context_length, type: GGUF_METADATA_VALUE_TYPE_UINT32
key: general.name, type: GGUF_METADATA_VALUE_TYPE_STRING
key: tokenizer.ggml.tokens, type: GGUF_METADATA_VALUE_TYPE_ARRAY
key: llama.embedding_length, type: GGUF_METADATA_VALUE_TYPE_UINT32
key: llama.feed_forward_length, type: GGUF_METADATA_VALUE_TYPE_UINT32
key: llama.attention.layer_norm_rms_epsilon, type:
GGUF_METADATA_VALUE_TYPE_FLOAT32 key: llama.rope.dimension_count, type:
GGUF_METADATA_VALUE_TYPE_UINT32 key: tokenizer.ggml.bos_token_id, type:
GGUF_METADATA_VALUE_TYPE_UINT32 key: llama.attention.head_count, type:
GGUF_METADATA_VALUE_TYPE_UINT32 key: tokenizer.ggml.scores, type:
GGUF_METADATA_VALUE_TYPE_ARRAY key: tokenizer.ggml.token_type, type:
GGUF_METADATA_VALUE_TYPE_ARRAY key: llama.block_count, type:
GGUF_METADATA_VALUE_TYPE_UINT32 key: llama.attention.head_count_kv, type:
GGUF_METADATA_VALUE_TYPE_UINT32 key: tokenizer.ggml.model, type:
GGUF_METADATA_VALUE_TYPE_STRING*/
  for (auto k : gguf.get_all_metadata_keys ())
    {
      fprintf (stderr, "key: %s, type: %s\n", k.first.c_str (),
               gguf.__gguf_type_name[k.second]);
    }

  {
    uint32_t vocab_size;
    gguf.get ("llama.vocab_size", vocab_size);
    fprintf (stderr, "llama.vocab_size: %u\n", vocab_size);
  }

  {
    std::string v;
    gguf.get ("general.architecture", v);
    fprintf (stderr, "general.architecture: %s\n", v.c_str ());
  }

  {
    uint32_t v = 0;
    gguf.get ("tokenizer.ggml.unknown_token_id", v);
    fprintf (stderr, "tokenizer.ggml.unknown_token_id: %u\n", v);
  }

  {
    uint32_t v = 0;
    gguf.get ("tokenizer.ggml.eos_token_id", v);
    fprintf (stderr, "tokenizer.ggml.eos_token_id: %u\n", v);
  }

  {
    uint32_t v = 0;
    gguf.get ("llama.embedding_length", v);
    fprintf (stderr, "llama.embedding_length: %u\n", v);
  }

  {
    auto tensor_infos = gguf.get_all_tensor_infos ();
    for (auto const &kv : tensor_infos)
      {
        fprintf (stderr,
                 "tensor name: %s, dtype: %s, shape = ", kv.first.c_str (),
                 gguf.__ggml_type_name[kv.second.dtype]);
        for (int i = 0; i < kv.second.ndim; ++i)
          {
            fprintf (stderr, "%lu ", kv.second.dims[kv.second.ndim - i - 1]);
          }
        fprintf (stderr, "\n");
      }
  }
#if 0
  {
    std::vector<std::string> toks;
    gguf.get ("tokenizer.ggml.tokens", toks);
    fprintf (stderr, "tokenizer.ggml.tokens: ");
    for (auto tok : toks)
      {
        fprintf (stderr, "%s\n", tok.c_str ());
      }
  }
#endif

#else
#endif
  return 0;
}
