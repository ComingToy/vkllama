#include "src/models/llama2.h"
#include <vector>
int
main (const int argc, const char *argv[])
{
  if (argc != 2)
    {
      fprintf (stderr, "usage: %s <path to checkpoitns>\n", argv[0]);
      return -1;
    }

  Model model;
  auto ret = model.init (argv[1]);
  if (ret != VK_SUCCESS)
    {
      fprintf (stderr, "failed at init model\n");
      return -1;
    }

  // std::vector<uint32_t> toks (128);
  // std::generate (toks.begin (), toks.end (),
  //                [x = uint32_t (0)] () mutable { return ++x; });

  std::vector<uint32_t> toks = { 2, 13709, 11823 };
  for (int i = 0; i < 10; ++i)
    {
      auto output = model (toks);
      toks.push_back (output.back ());
    }

  for (auto tok : toks)
    {
      std::cerr << tok << " ";
    }
  std::cerr << std::endl;
  return 0;
}

