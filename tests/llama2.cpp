#define USE_GGUF 1
#if USE_GGUF
#include "models/llama2_gguf.h"
extern "C"
{
#include "gguflib.h"
}
#else
#include "models/llama2.h"
#endif
#include "sentencepiece_processor.h"
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

static std::unique_ptr<llama2::Variables>
load_checkpoint_file (std::string const &fname)
{
  int fd = open (fname.c_str (), O_RDONLY);
  if (fd < 0)
    {
      char buf[512];
      strerror_r (errno, buf, sizeof (buf));
      fprintf (stderr, "open checkpoitn %s failed: %s\n", fname.c_str (), buf);
      return nullptr;
    }

  auto variables = std::make_unique<llama2::Variables> ();
  if (!variables->ParseFromFileDescriptor (fd))
    {
      fprintf (stderr, "parse checkpoint %s failed.\n", fname.c_str ());
      return nullptr;
    }

  return variables;
}

static std::vector<std::unique_ptr<llama2::Variables> >
load_checkpoint (std::string const &path)
{
  DIR *dir = opendir (path.c_str ());
  if (!dir)
    {
      fprintf (stderr, "open dir %s failed\n", path.c_str ());
      return {};
    }

  std::vector<std::string> blocks;
  struct dirent *d;
  while ((d = readdir (dir)))
    {
      if (d->d_type != DT_REG)
        continue;
      blocks.push_back (path + "/" + d->d_name);
    }

  auto ret = closedir (dir);
  if (ret)
    {
      fprintf (stderr, "close %s dir failed: %d\n", path.c_str (), ret);
      return {};
    }

  std::vector<std::unique_ptr<llama2::Variables> > checkpoint;
  for (auto const &block : blocks)
    {
      auto ckpt = load_checkpoint_file (block);
      if (!ckpt)
        {
          fprintf (stderr, "load checkpoint %s failed\n", block.c_str ());
          return {};
        }

      checkpoint.push_back (std::move (ckpt));
    }

  return checkpoint;
}

int
main (const int argc, const char *argv[])
{
  if (argc != 5)
    {
      fprintf (stderr,
               "usage: %s <path to checkpoitns> <path to bpe> <enable kv "
               "cache> <prompt>\n",
               argv[0]);
      return -1;
    }

  sentencepiece::SentencePieceProcessor sp;
  auto status = sp.Load (argv[2]);
  if (!status.ok ())
    {
      fprintf (stderr, "init bpe failed: %s\n", status.ToString ().c_str ());
      return static_cast<int> (status.code ());
    }

  std::string buffer = argv[4];
  string_process_escapes (buffer);

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

#if !USE_GGUF
  std::unordered_map<std::string, const llama2::Variable *> state_dict;
  auto checkpoint = load_checkpoint (argv[1]);
  if (checkpoint.empty ())
    {
      return -1;
    }
  for (auto &variables : checkpoint)
    {
      for (const auto &var : variables->variables ())
        {
          fprintf (stderr, "load variable %s from checkpoint\n",
                   var.name ().c_str ());
          state_dict[var.name ()] = &var;
        }
    }
#else
  auto *gguf = gguf_open (argv[1]);
  if (!gguf)
    {
      fprintf (stderr, "output gguf file failed.\n");
    }

  /* Show all the key-value pairs. */
  gguf_key key;
  while (gguf_get_key (gguf, &key))
    {
      printf ("%.*s: [%s] ", (int)key.namelen, key.name,
              gguf_get_value_type_name (key.type));
      gguf_print_value (gguf, key.type, key.val, 0);
      printf ("\n");
    }

  auto *state_dict = gguf;

#endif

  vkllama::Model model;
  auto ret = model.init (state_dict);

  if (ret != VK_SUCCESS)
    {
      fprintf (stderr, "failed at init model\n");
      return -1;
    }

  for (int r = 0; r < 1; ++r)
    {
      std::vector<uint32_t> toks;
      std::transform (
          prompt.cbegin (), prompt.cend (), std::back_inserter (toks),
          [] (const int tok) { return static_cast<uint32_t> (tok); });

      auto init_out = model (toks, 0);
      toks.push_back (init_out.back ());

      int enable_kvcache = ::atoi (argv[3]);
      for (int i = 1; i < 100; ++i)
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

      std::vector<int> output;
      std::transform (
          toks.cbegin (), toks.cend (), std::back_inserter (output),
          [] (uint32_t const tok) { return static_cast<int> (tok); });

      std::string content;
      sp.Decode (output, &content);
      std::cerr << "prompt: " << argv[4] << std::endl
                << "output: " << content << std::endl;
    }

  return 0;
}

