#include "models/llama2.h"
#include "models/samplers.h"
#include "models/tokenizer.h"
#include "sentencepiece_processor.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <errno.h>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

struct Params
{
  std::string model_file;
  std::string tokenizer_file;
  std::string system_message;
  std::vector<std::string> anti_prompt;
  std::string sampler;
  struct
  {
    int topk;
    float p;
  } sampler_option;
};

#define _H(s) "\033[1m" #s "\033[0m"

static void
show_usage (int argc, char *const argv[])
{
  // clang-format off
  const char *fmt =
_H (NAME)"\n"
"    chat - chat with llama2 interactive \n\n"
_H (SYNOPSIS)"\n"
"    chat " _H(-m) " path " _H(-e) " path " _H(-t) " path [" _H(-s) " {top_k|top_p}] [" _H(-k) " value] " "[" _H(-p) " value]" "\n"
"\n"
_H(DESCRIPTION)"\n"
"    the options are follow:\n"
"    " _H(-m) "\tpath to the gguf model file\n"
"    " _H(-e) "\tpath to the tokenizer model file\n"
"    " _H(-t) "\tsystem message filled into prompt template\n"
"    " _H(-s) "\tsampler. top_k or top_p are supported. (default: top_k)\n"
"    " _H(-k) "\tthe k option of top_k sampler. (default: 40)\n"
"    " _H(-p) "\tthe p option of top_p sampler. (default: 0.75)\n"
;
  // clang-format on
  fprintf (stdout, fmt);
}

static int
parse_params_from_cmdline (int argc, char *const argv[], Params *params)
{
  int ch = -1;
  while ((ch = ::getopt (argc, argv, "m:t:a:s:k:p:e:")) != -1)
    {
      switch (ch)
        {
        case 'm':
          params->model_file = optarg;
          break;
        case 'e':
          params->tokenizer_file = optarg;
          break;
        case 't':
          params->system_message = optarg;
          break;
        case 'a':
          params->anti_prompt.push_back (optarg);
          break;
        case 's':
          params->sampler = optarg;
          break;
        case 'k':
          params->sampler_option.topk = ::atoi (optarg);
          break;
        case 'p':
          params->sampler_option.p = ::atof (optarg);
          break;
        case '?':
        default:
          show_usage (argc, argv);
          return -1;
        }
    }

  return 0;
}

static void
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
    }
}

static int
load_prompt_template (const char *path, std::string &content)
{
  auto fs = fopen (path, "rb");
  if (!fs)
    {
      fprintf (stderr, "failed at open prompt template file: %s\n",
               strerror (errno));
      return -1;
    }

  int n = 0;
  char buf[1024];
  int ret = 0;
  while ((n = fread (buf, 1, sizeof (buf), fs)) >= 0)
    {
      if (n > 0)
        {
          content.append (buf, n);
        }

      if (n < sizeof (buf))
        {
          if (feof (fs))
            {
              ret = 0;
            }
          else if (ferror (fs))
            {
              fprintf (stderr, "failed at read prompt template file: %s\n",
                       strerror (errno));
              ret = -1;
            }
          else
            {
              continue;
            }
          goto out;
        }
    }

out:
  fclose (fs);
  return ret;
}

static bool
is_anti_prompt (std::deque<std::string> const &output_buf,
                std::string const &anti, std::string &removed_anti)
{
  std::string content;
  for (auto &piece : output_buf)
    {
      content.append (piece);
    }

  if (content.size () < anti.size ())
    {
      return false;
    }

  auto pos = content.find (anti);
  if (pos == std::string::npos)
    {
      return false;
    }

  removed_anti = content.substr (0, pos);
  return true;
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

struct Session
{
  std::vector<int> toks;
  std::deque<std::string> output_buf;
  int offset;
};

int
wait_for_input (std::string &line)
{
  std::getline (std::cin, line);
  return 0;
}

// session stage ref: https://gpus.llm-utils.org/llama-2-prompt-template/
static std::string prompt_template
    = R"([INST]<<SYS>>{system_message}<</SYS>>{user_message}[/INST])";

int
start_sess (const Params &params, vkllama::Model &model,
            ::sentencepiece::SentencePieceProcessor &sp)
{
  Session sess = { {}, {}, 0 };

  TopkSampler sampler (40);

  // if (std::string ("top_k") == params.sampler)
  //   {
  //     sampler = std::make_shared<TopkSampler> (params.sampler_option.topk);
  //   }
  // else
  //   {
  //     sampler = std::make_shared<TopPSampler> (params.sampler_option.p);
  //   }

  fflush (stdout);
  while (true)
    {
      fprintf (stdout, "[INFO]: total %zu toks output.\n", sess.toks.size ());
      fprintf (stdout, "\n[[USER]]: ");
      fflush (stdout);

      std::string line;
      wait_for_input (line);
      if (line.empty ())
        {
          continue;
        }

      auto temp = sess.offset == 0 ? prompt_template
                                   : "[INST] {user_message} [/INST]";
      replace_all (temp, "{user_message}", line);
      line = temp;

      fprintf (stdout, "[[AI]]: ");
      fflush (stdout);
      {
        std::vector<int> toks;
        sp.Encode (line, &toks);
        sess.toks.push_back (sp.bos_id ());
        sess.toks.insert (sess.toks.end (), toks.cbegin (), toks.cend ());
      }

      int next_token_id = sp.unk_id ();
      {
        std::vector<uint32_t> inp;
        std::transform (sess.toks.cbegin () + sess.offset, sess.toks.cend (),
                        std::back_inserter (inp),
                        [] (auto v) { return (uint32_t)v; });

        auto logits = model (inp, std::max (sess.offset - 1, 0));
        if (!logits.ok ())
          {
            std::cerr << "model infer failed: " << logits.status ()
                      << std::endl;
            return -1;
          }

        auto dim = logits->size ();
        next_token_id = sampler.sample (logits->data (), dim);
        sess.offset += inp.size ();
      }

      while (true)
        {
          if (next_token_id == sp.eos_id ())
            {
              std::string content;
              for (auto &piece : sess.output_buf)
                {
                  content.append (piece);
                }
              fprintf (stdout, "%s[end of text]", content.c_str ());
              fflush (stdout);
              sess.output_buf.clear ();
              break;
            }

          auto piece = sp.IdToPiece (next_token_id);
          if (sp.IsByte (next_token_id))
            {
              piece = escape_byte (piece);
            }

          replace_all (piece, "▁", " ");

          sess.output_buf.push_back (piece);
          fprintf (stdout, "%s", piece.c_str ());
          fflush (stdout);
#if 1
          bool is_anti = false;
          std::string removed_anti;
          for (auto anti : params.anti_prompt)
            {
              if (is_anti_prompt (sess.output_buf, anti, removed_anti))
                {
                  is_anti = true;
                  sess.output_buf.clear ();
                  break;
                }
            }

          if (is_anti)
            {
              break;
            }
#endif

          auto logits = model ({ (uint32_t)next_token_id },
                               std::max (sess.offset - 1, 0));
          next_token_id = sampler.sample (logits->data (), logits->size ());
          sess.offset += 1;
          sess.toks.push_back (next_token_id);
        }
    }

  return 0;
}

int
main (int argc, char *const argv[])
{
  int ret = 0;

  Params params = { .model_file = "",
                    .tokenizer_file = "",
                    .system_message = "",
                    .anti_prompt = {},
                    .sampler = "top_k",
                    .sampler_option = { .topk = 10, .p = 0.9 } };

  if ((ret = parse_params_from_cmdline (argc, argv, &params)) != 0)
    {
      return ret;
    }

  fprintf (stderr,
           "model_file: %s\nsystem message: %s\nsampler: %s\ntopk: %d\np:%f\n",
           params.model_file.c_str (), params.system_message.c_str (),
           params.sampler.c_str (), params.sampler_option.topk,
           params.sampler_option.p);

  for (size_t i = 0; i < params.anti_prompt.size (); ++i)
    {
      fprintf (stderr, "anti_prompt[%zu]: %s\n", i,
               params.anti_prompt[i].c_str ());
    }

  auto gguf = gguf_open (params.model_file.c_str ());
  if (!gguf)
    {
      fprintf (stderr, "open model file %s failed.\n",
               params.model_file.c_str ());
      return -1;
    }

  std::map<std::string, gguf_key> meta;
  std::map<std::string, gguf_tensor> tensors;
  read_gguf (gguf, meta, tensors);

  sentencepiece::SentencePieceProcessor sp;
  // auto s = load_tokenizer (sp, meta);
  auto s = sp.Load (params.tokenizer_file.c_str ());
  if (!s.ok ())
    {
      fprintf (stderr, "load tokenizer failed: %s\n", s.ToString ().c_str ());
      return -1;
    }

  vkllama::Model model;
  if (auto s = model.init (meta, tensors); !s.ok ())
    {
      std::cerr << "failed at model init: " << s << std::endl;
      return -1;
    }

  gguf_close (gguf);

  fflush (stderr);

  replace_all (prompt_template, "{system_message}", params.system_message);

  start_sess (params, model, sp);
  return 0;
}
