#include "gguf.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

uint64_t __gguf_elem_size[13] = { sizeof (uint8_t),
                                  sizeof (int8_t),
                                  sizeof (uint16_t),
                                  sizeof (int16_t),
                                  sizeof (uint32_t),
                                  sizeof (int32_t),
                                  sizeof (float),
                                  sizeof (bool),
                                  0,
                                  0,
                                  sizeof (uint64_t),
                                  sizeof (int64_t),
                                  sizeof (double) };

gguf_header_t *
map_gguf_file (int fd)
{
  char err_msg[512];
  struct stat stat;
  auto ret = ::fstat (fd, &stat);
  if (ret < 0)
    {
      fprintf (stderr, "fstat failed: %s\n",
               ::strerror_r (errno, err_msg, sizeof (err_msg)));
      return nullptr;
    }

  fprintf (stderr, "fstat file size: %lu\n", stat.st_size);

  auto *mapped_data
      = ::mmap (nullptr, stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped_data == MAP_FAILED)
    {
      fprintf (stderr, "mapped failed: %s\n",
               ::strerror_r (errno, err_msg, sizeof (err_msg)));
      return nullptr;
    }

  return reinterpret_cast<gguf_header_t *> (mapped_data);
}
