#include "models/llama2.h"
#include "models/tokenizer.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <iterator>
#include <memory>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

void
string_process_escapes (std::string &input)
{
  std::size_t input_len = input.length ();
  std::size_t output_idx = 0;

  for (std::size_t input_idx = 0; input_idx < input_len; ++input_idx)
    {
      if (input[input_idx] == '\\' && input_idx + 1 < input_len)
        {
          switch (input[++input_idx])
            {
            case 'n':
              input[output_idx++] = '\n';
              break;
            case 'r':
              input[output_idx++] = '\r';
              break;
            case 't':
              input[output_idx++] = '\t';
              break;
            case '\'':
              input[output_idx++] = '\'';
              break;
            case '\"':
              input[output_idx++] = '\"';
              break;
            case '\\':
              input[output_idx++] = '\\';
              break;
            case 'x':
              // Handle \x12, etc
              if (input_idx + 2 < input_len)
                {
                  const char x[3]
                      = { input[input_idx + 1], input[input_idx + 2], 0 };
                  char *err_p = nullptr;
                  const long val = std::strtol (x, &err_p, 16);
                  if (err_p == x + 2)
                    {
                      input_idx += 2;
                      input[output_idx++] = char (val);
                      break;
                    }
                }
              // fall through
            default:
              input[output_idx++] = '\\';
              input[output_idx++] = input[input_idx];
              break;
            }
        }
      else
        {
          input[output_idx++] = input[input_idx];
        }
    }

  input.resize (output_idx);
}

void
print_sp_model (sentencepiece::ModelProto const &model)
{
  fprintf (
      stderr,
      "size of model::pieces = %d\nhas trainer_spec = %d\nhas normalizer_spec "
      "= %d\n has self_test_data = %d\nhas denormalizer_spec = %d\n",
      model.pieces_size (), (int)model.has_trainer_spec (),
      (int)model.has_normalizer_spec (), (int)model.has_self_test_data (),
      (int)model.has_denormalizer_spec ());

  for (int i = 0; i < std::max (10, model.pieces_size ()); ++i)
    {
      auto piece = model.pieces (i);
      fprintf (stderr, "index %d, piece = %s, score = %f, type = %d\n", i,
               piece.has_piece () ? piece.piece ().c_str () : "null",
               piece.score (), piece.type ());
    }
}

int
main (const int argc, const char *argv[])
{
  if (argc != 4)
    {
      fprintf (stderr,
               "usage: %s <path to checkpoitns> <enable kv "
               "cache> <prompt>\n",
               argv[0]);
      return -1;
    }

  std::string buffer = argv[3];
  string_process_escapes (buffer);

  auto *gguf = gguf_open (argv[1]);
  if (!gguf)
    {
      fprintf (stderr, "output gguf file failed.\n");
    }

  /* Show all the key-value pairs. */
  std::map<std::string, gguf_key> gguf_kv;
  gguf_key key;
  while (gguf_get_key (gguf, &key))
    {
      printf ("%.*s: [%s] ", (int)key.namelen, key.name,
              gguf_get_value_type_name (key.type));
      gguf_print_value (gguf, key.type, key.val, 0);
      printf ("\n");

      std::string name (key.name, key.namelen);
      gguf_kv[name] = key;
    }

  std::map<std::string, gguf_tensor> tensors;

  gguf_tensor tensor;
  while (gguf_get_tensor (gguf, &tensor))
    {
      std::string name (tensor.name, tensor.namelen);
      tensors[name] = tensor;
    }

  sentencepiece::SentencePieceProcessor sp;
  auto s = load_tokenizer (sp, gguf_kv);
  if (s.code () != sentencepiece::util::StatusCode::kOk)
    {
      fprintf (stderr, "load tokenizer failed: %s\n", s.ToString ().c_str ());
      return -1;
    }

  std::vector<int> prompt_tmp;
  sp.Encode (buffer, &prompt_tmp);
  std::cerr << "input tokens: ";
  for (auto t : prompt_tmp)
    {
      std::cerr << t << " ";
    }
  std::cerr << std::endl;

  std::vector<int> prompt = { sp.bos_id () };
  std::copy (prompt_tmp.cbegin (), prompt_tmp.cend (),
             std::back_inserter (prompt));

  vkllama::Model model;
  auto ret = model.init (gguf_kv, tensors);

  if (ret != VK_SUCCESS)
    {
      fprintf (stderr, "failed at init model\n");
      return -1;
    }

  fprintf (stderr, "all weights are uploaded to device\n");
  for (int r = 0; r < 1; ++r)
    {
      std::vector<uint32_t> toks;
      std::transform (
          prompt.cbegin (), prompt.cend (), std::back_inserter (toks),
          [] (const int tok) { return static_cast<uint32_t> (tok); });

      auto init_out = model (toks, 0);
      toks.push_back (init_out.back ());

      fprintf (stderr, "prompt tokens are generated\n");

      int enable_kvcache = ::atoi (argv[2]);
      auto t0 = std::chrono::high_resolution_clock::now ();
      for (int i = 1; i < 200; ++i)
        {
          auto output = enable_kvcache
                            ? model ({ toks.back () }, toks.size () - 1)
                            : model (toks, 0);
          if ((int)output.back () == sp.eos_id ()
              || sp.bos_id () == (int)output.back ())
            {
              break;
            }
          toks.push_back (output.back ());
          fprintf (stderr, "output %d tokens\n", i);
        }
      auto t1 = std::chrono::high_resolution_clock::now ();
      auto milliseconds
          = std::chrono::duration_cast<std::chrono::milliseconds> (t1 - t0)
                .count ();
      std::cerr << "infer speed: " << toks.size () * 1000.0f / milliseconds
                << " tokens/s" << std::endl;

      std::vector<int> output;
      std::transform (
          toks.cbegin (), toks.cend (), std::back_inserter (output),
          [] (uint32_t const tok) { return static_cast<int> (tok); });

      std::string content;
      sp.Decode (output, &content);
      std::cerr << "prompt: " << argv[3] << std::endl
                << "output: " << content << std::endl;
    }

  return 0;
}

