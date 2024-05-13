#include "models/gguf/gguf.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>

int
main (const int argc, const char *argv[])
{
  if (argc != 2)
    {
      fprintf (stderr, "usage %s <path to gguf>\n", argv[0]);
      return -1;
    }

  fprintf (stderr, "input gguf file: %s\n", argv[1]);
  int fd = ::open (argv[1], O_RDONLY);
  if (fd < 0)
    {
      char buf[512];
      fprintf (stderr, "open file failed: %s\n",
               strerror_r (errno, buf, sizeof (buf)));
    }
  auto header = map_gguf_file (fd);
  fprintf (
      stderr,
      "magic: 0x%x\nversion: %u\ntensor_count: %lu\nmetadata_kv_count: %lu\n",
      header->magic, header->version, header->tensor_count,
      header->metadata_kv_count);

  struct gguf_metadata_kv_t *metadata = header->metadata_kv;
  for (int i = 0; i < header->metadata_kv_count; ++i)
    {
      gguf_string_t *key = &metadata->key;
      auto value_payload = (void *)(reinterpret_cast<uint8_t *> (metadata)
                                    + sizeof (gguf_string_t) + key->len);
      auto value_type = *(gguf_metadata_value_type *)value_payload;

      auto *values
          = (gguf_metadata_value_t *)((uint8_t *)value_payload
                                      + sizeof (gguf_metadata_value_type));

      if (value_type != GGUF_METADATA_VALUE_TYPE_ARRAY)
        {
          metadata = (gguf_metadata_kv_t *)((uint8_t *)values
                                            + sizeof (values->value.string)
                                            + values->value.string.len);

          if (value_type == GGUF_METADATA_VALUE_TYPE_STRING)
            {
              fprintf (
                  stderr, "meta data key: %.*s, type: string, value: %.*s\n",
                  (int)key->len, key->string, (int)values->value.string.len,
                  values->value.string.string);
            }
          else if (value_type == GGUF_METADATA_VALUE_TYPE_BOOL)
            {
              fprintf (stderr, "meta data key: %.*s, type: bool, value: %d\n",
                       (int)key->len, key->string, (int)values->value.bool_);
            }
          else if (value_type == GGUF_METADATA_VALUE_TYPE_FLOAT32)
            {
              fprintf (stderr,
                       "meta data key: %.*s, type: float32, value: %f\n",
                       (int)key->len, key->string, values->value.float32);
            }
          else if (value_type == GGUF_METADATA_VALUE_TYPE_FLOAT64)
            {
              fprintf (stderr,
                       "meta data key: %.*s, type: float64, value: %lf\n",
                       (int)key->len, key->string, values->value.float64);
            }
          else if (value_type == GGUF_METADATA_VALUE_TYPE_INT16)
            {
              fprintf (stderr, "meta data key: %.*s, type: int16, value: %d\n",
                       (int)key->len, key->string, (int)values->value.int16);
            }
          else if (value_type == GGUF_METADATA_VALUE_TYPE_INT32)
            {
              fprintf (stderr, "meta data key: %.*s, type: int32, value: %d\n",
                       (int)key->len, key->string, (int)values->value.int32);
            }
          else if (value_type == GGUF_METADATA_VALUE_TYPE_INT8)
            {
              fprintf (stderr, "meta data key: %.*s, type: int8, value: %d\n",
                       (int)key->len, key->string, (int)values->value.int8);
            }
          else if (value_type == GGUF_METADATA_VALUE_TYPE_INT64)
            {
              fprintf (stderr,
                       "meta data key: %.*s, type: int64, value: %ld\n",
                       (int)key->len, key->string, values->value.int64);
            }
          else if (value_type == GGUF_METADATA_VALUE_TYPE_UINT8)
            {
              fprintf (stderr, "meta data key: %.*s, type: uint8, value: %d\n",
                       (int)key->len, key->string, (int)values->value.uint8);
            }
          else if (value_type == GGUF_METADATA_VALUE_TYPE_UINT16)
            {
              fprintf (stderr,
                       "meta data key: %.*s, type: uint16, value: %d\n",
                       (int)key->len, key->string, (int)values->value.uint16);
            }

          else if (value_type == GGUF_METADATA_VALUE_TYPE_UINT32)
            {
              fprintf (stderr,
                       "meta data key: %.*s, type: uint32, value: %u\n",
                       (int)key->len, key->string, values->value.uint32);
            }
          else if (value_type == GGUF_METADATA_VALUE_TYPE_UINT32)
            {
              fprintf (stderr,
                       "meta data key: %.*s, type: uint64, value: %lu\n",
                       (int)key->len, key->string, values->value.uint64);
            }

          size_t offset = 0;
          if (value_type == GGUF_METADATA_VALUE_TYPE_STRING)
            {
              offset
                  = sizeof (values->value.string) + values->value.string.len;
            }
          else
            {
              offset = __gguf_elem_size[value_type];
            }
          metadata = (struct gguf_metadata_kv_t *)((uint8_t *)values + offset);
        }
      else
        {
          auto *array = &values->array;
          fprintf (
              stderr,
              "meta data key: %.*s, type: array, elem type: %u, len = %lu\n",
              (int)key->len, key->string, array->type, array->len);

          size_t offset = 0;
          if (array->type == GGUF_METADATA_VALUE_TYPE_STRING)
            {
              gguf_string_t *elems = (gguf_string_t *)array->array;
              for (int i = 0; i < array->len; ++i)
                {
                  size_t elem_size = sizeof (*elems) + elems->len;
                  offset += elem_size;
                  elems = (gguf_string_t *)((uint8_t *)elems + elem_size);
                }
            }
          else
            {
              offset = __gguf_elem_size[array->type] * array->len;
            }

          metadata = (struct gguf_metadata_kv_t *)((uint8_t *)values
                                                   + sizeof (values->array)
                                                   + offset);
        }
    }
  return 0;
}
