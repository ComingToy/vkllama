#include "models/llama2.h"
#include "sentencepiece_processor.h"
#include <cstdio>
#include <iterator>
#include <unistd.h>
#include <vector>

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

  GGUF gguf (argv[1]);
  if (gguf.init () != 0)
    {
      fprintf (stderr, "failed at init gguf\n");
      return -1;
    }

  sentencepiece::SentencePieceProcessor sp;
  auto status = sp.Load (argv[2]);
  if (!status.ok ())
    {
      fprintf (stderr, "init bpe failed: %s\n", status.ToString ().c_str ());
      return static_cast<int> (status.code ());
    }

  std::vector<int> prompt_;
  sp.Encode (argv[3], &prompt_);
  std::cerr << "input tokens: ";
  // std::vector<int> prompt = { 1,    1029, 29537, 1200, 325,   268,
  //                             5242, 6848, 29584, 13,   29530, 29537 };
  std::vector<int> prompt = { 1 };
  prompt.insert (prompt.end (), prompt_.begin (), prompt_.end ());
  for (auto t : prompt)
    {
      std::cerr << t << " ";
    }
  std::cerr << std::endl;

  // std::vector<uint32_t> toks (128);
  // std::generate (toks.begin (), toks.end (),
  //                [x = uint32_t (0)] () mutable { return ++x; });

  Model model;
  auto ret = model.init (gguf);
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
      for (int i = 0; i < 10; ++i)
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

      std::cerr << "output toks: ";
      for (auto tok : output)
        {
          std::cerr << tok << " ";
        }
      std::cerr << std::endl;
    }

  return 0;
}

