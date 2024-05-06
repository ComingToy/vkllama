#include "src/models/llama2.h"
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

  auto output = model ({ 1, 2, 3, 4, 5 });
  fprintf (stdout, "output size: %zu\n", output.size ());
  return 0;
}

