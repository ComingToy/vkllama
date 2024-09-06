#include "models/llama2.h"
#include "models/samplers.h"
#include "models/tokenizer.h"
#include "sentencepiece.pb.h"
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

static std::string prompt_template = R"(
User: Hello, Bob.
Bob: Hello. How may I help you today?
User: Please tell me the largest city in Europe.
Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.
User: )";

static bool is_anti_prompt(std::string const& output_buf, std::string const& anti)
{
  if (output_buf.size () < anti.size ())
    {
      return false;
    }

  auto start = output_buf.size () - anti.size ();
  return output_buf.substr (start, anti.size ()) == anti;
}
static void
replace_all (std::string &s, const std::string &search,
             const std::string &replace)
{
  std::string result;
  for (size_t pos = 0;; pos += search.length ())
    {
      auto new_pos = s.find (search, pos);
      if (new_pos == std::string::npos)
        {
          result += s.substr (pos, s.size () - pos);
          break;
        }
      result += s.substr (pos, new_pos - pos) + replace;
      pos = new_pos;
    }
  s = std::move (result);
}

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
  if (argc != 3)
    {
      fprintf (stderr, "usage: %s <path to checkpoitns> <prompt>\n", argv[0]);
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
  auto s = load_tokenizer (sp, gguf_kv);
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

  std::vector<int> prompt_tmp;
  std::string buffer = argv[2];
  // replace_all (buffer, " ", "\xe2\x96\x81");
  // string_process_escapes (buffer);
  buffer = prompt_template + buffer;

  sp.Encode (buffer, &prompt_tmp);

  std::vector<int> prompt = { sp.bos_id () };
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

  if (ret != VK_SUCCESS)
    {
      fprintf (stderr, "failed at init model\n");
      return -1;
    }

  fprintf (stderr, "all weights are uploaded to device\n");

  std::unique_ptr<TopkSampler> samplers (new TopkSampler (20));

  for (int r = 0; r < 1; ++r)
    {
      std::vector<uint32_t> prompt_inp;
      std::transform (
          prompt.cbegin (), prompt.cend (), std::back_inserter (prompt_inp),
          [] (const int tok) { return static_cast<uint32_t> (tok); });

      auto t0 = std::chrono::high_resolution_clock::now ();
      auto init_out = model (prompt_inp, 0);
      auto t1 = std::chrono::high_resolution_clock::now ();
      size_t candidate_size = init_out.size () / prompt_inp.size ();

      std::vector<int> toks (prompt);

      toks.push_back (samplers->sample (init_out.data () + init_out.size ()
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
      for (int i = 1; i < 8192; ++i)
        {
          auto output = model ({ (uint32_t)toks.back () }, toks.size ());
          toks.push_back (samplers->sample (output.data (), output.size ()));

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

          replace_all (piece, "▁", " ");
          std::cerr << piece;
          output_buf.append (piece);

          if ((int)toks.back () == sp.eos_id ()
              || sp.bos_id () == (int)toks.back ()
              || toks.back () == eot_token_id)
            {
              std::cerr << "[end of text]" << std::endl;
              // break;
            }

          if (toks.size () >= 4096)
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

