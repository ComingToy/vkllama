#include "models/llama2.h"
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
  if (argc != 4)
    {
      fprintf (stderr,
               "usage: %s <path to checkpoitns> <path to bpe> <prompt>\n",
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

  std::vector<int> prompt;
  sp.Encode (std::string (argv[3]), &prompt);
  std::cerr << "input tokens: ";
  for (auto t : prompt)
    {
      std::cerr << t << " ";
    }
  std::cerr << std::endl;

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
      for (int i = 0; i < 20; ++i)
        {
          auto output = model (toks);
          if ((int)output.back () == sp.eos_id ())
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
      std::cerr << "prompt: " << argv[3] << std::endl
                << "output: " << content << std::endl;
    }

  return 0;
}

