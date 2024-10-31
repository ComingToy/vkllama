#include "absl/strings/str_replace.h"
#include "models/llama2.h"
#include "models/samplers.h"
#include "models/tokenizer.h"
#include "sentencepiece.pb.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
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

std::string
escape_byte (std::string const &b)
{
  auto pos = b.find ("0x");
  std::string result;
  if (pos != std::string::npos)
    {
      char v = (char)std::strtol (b.substr (pos).c_str (), NULL, 16);
      result.push_back (v);
      return result;
    }
  return b;
}

int
main (const int argc, const char *argv[])
{
  if (argc != 4)
    {
      fprintf (stderr,
               "usage: %s <path to checkpoitns> <predict tokens> <prompt>\n",
               argv[0]);
      return -1;
    }

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
  auto s = sp.Load ("/home/conley/github/llama2-3b/tokenizer.model");
  // auto s = load_tokenizer (sp, gguf_kv);
  if (s.code () != sentencepiece::util::StatusCode::kOk)
    {
      fprintf (stderr, "load tokenizer failed: %s\n", s.ToString ().c_str ());
      return -1;
    }

  int eot_token_id = -1;
  for (uint32_t i = 0; i < sp.GetPieceSize (); ++i)
    {
      auto piece = sp.IdToPiece (i);
      if (
          // TODO: gemma "<end_of_turn>" is exported as a normal
          // token, so the following check does not work
          //       need to fix convert script
          // vocab.id_to_token[t.second].type ==
          // LLAMA_TOKEN_TYPE_CONTROL &&
          (piece == "<|eot_id|>" || piece == "<|im_end|>" || piece == "<|end|>"
           || piece == "<end_of_turn>" || piece == "<|endoftext|>"))
        {
          eot_token_id = i;
          fprintf (stderr, "use eot token id = %d\n", eot_token_id);
          break;
        }
    }

  auto pred_tokens = ::atoi (argv[2]);
  std::vector<int> prompt_tmp;
  std::string buffer = argv[3];

  sp.Encode (buffer, &prompt_tmp);

  std::vector<int> prompt;
  std::copy (prompt_tmp.cbegin (), prompt_tmp.cend (),
             std::back_inserter (prompt));

  std::cerr << "input tokens: ";
  for (auto t : prompt)
    {
      std::cerr << t << " ";
    }
  std::cerr << std::endl;

  vkllama::Model model;
  auto ret = model.init (gguf_kv, tensors);

  if (!ret.ok ())
    {
      fprintf (stderr, "failed at init model: %s\n", ret.ToString ().c_str ());
      return -1;
    }

  fprintf (stderr, "all weights are uploaded to device\n");

  std::unique_ptr<TopkSampler> samplers (new TopkSampler (1));

  for (int r = 0; r < 1; ++r)
    {
      std::vector<uint32_t> prompt_inp;
      std::transform (
          prompt.cbegin (), prompt.cend (), std::back_inserter (prompt_inp),
          [] (const int tok) { return static_cast<uint32_t> (tok); });

      auto t0 = std::chrono::high_resolution_clock::now ();
      auto init_out = model (prompt_inp, 0);
      if (!init_out.ok ())
        {
          std::cerr << "model infer failed: " << init_out.status ()
                    << std::endl;
          return -1;
        }
      auto t1 = std::chrono::high_resolution_clock::now ();
      size_t candidate_size = init_out->size () / prompt_inp.size ();

      std::vector<int> toks (prompt);

      toks.push_back (samplers->sample (init_out->data () + init_out->size ()
                                            - candidate_size,
                                        candidate_size));

      auto milliseconds
          = std::chrono::duration_cast<std::chrono::milliseconds> (t1 - t0)
                .count ();
      fprintf (stderr,
               "prompt tokens are generated. prompt speed: %f tokens/s\n",
               prompt.size () * 1000.f / milliseconds);

      std::cerr << buffer;

      auto t2 = std::chrono::high_resolution_clock::now ();
      std::string output_buf;
      for (int i = 1; i < pred_tokens; ++i)
        {
          auto output = model ({ (uint32_t)toks.back () }, toks.size ());
          if (!output.ok ())
            {
              std::cerr << "model infer failed: " << output.status ()
                        << std::endl;
            }
          toks.push_back (samplers->sample (output->data (), output->size ()));

          auto piece = sp.IdToPiece (toks.back ());

          if (sp.IsByte (toks.back ()))
            {
              piece = escape_byte (piece);
            }

          if (piece == "<0x09>")
            {
              piece = "\t";
            }

          if (sp.IsControl (toks.back ()))
            {
              piece = piece + "[CONTROL]";
            }

          piece = absl::StrReplaceAll (piece, { { "▁", " " } });
          std::cerr << piece;
          output_buf.append (piece);

          if ((int)toks.back () == sp.eos_id ()
              || sp.bos_id () == (int)toks.back ()
              || toks.back () == eot_token_id)
            {
              std::cerr << "[end of text]" << std::endl;
            }

          if (toks.size () == 4096)
            {
              fprintf (stderr, "\n\ncontext overflow.!!!!!!!!!!\n\n");
            }
        }

      auto t3 = std::chrono::high_resolution_clock::now ();
      milliseconds
          = std::chrono::duration_cast<std::chrono::milliseconds> (t3 - t2)
                .count ();
      std::cerr << "eval speed: "
                << (toks.size () - prompt.size ()) * 1000.0f / milliseconds
                << " tokens/s" << std::endl;
    }

  return 0;
}

