#ifndef __VKLLAMA_GGUF_H__
#define __VKLLAMA_GGUF_H__
#include <algorithm>
#include <errno.h>
#include <fcntl.h>
#include <iterator>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#define __PACKED __attribute__ ((packed))
enum ggml_type : uint32_t
{
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_1 = 3,
  // GGML_TYPE_Q4_2 = 4, support has been removed
  // GGML_TYPE_Q4_3 = 5, support has been removed
  GGML_TYPE_Q5_0 = 6,
  GGML_TYPE_Q5_1 = 7,
  GGML_TYPE_Q8_0 = 8,
  GGML_TYPE_Q8_1 = 9,
  GGML_TYPE_Q2_K = 10,
  GGML_TYPE_Q3_K = 11,
  GGML_TYPE_Q4_K = 12,
  GGML_TYPE_Q5_K = 13,
  GGML_TYPE_Q6_K = 14,
  GGML_TYPE_Q8_K = 15,
  GGML_TYPE_IQ2_XXS = 16,
  GGML_TYPE_IQ2_XS = 17,
  GGML_TYPE_IQ3_XXS = 18,
  GGML_TYPE_IQ1_S = 19,
  GGML_TYPE_IQ4_NL = 20,
  GGML_TYPE_IQ3_S = 21,
  GGML_TYPE_IQ2_S = 22,
  GGML_TYPE_IQ4_XS = 23,
  GGML_TYPE_I8 = 24,
  GGML_TYPE_I16 = 25,
  GGML_TYPE_I32 = 26,
  GGML_TYPE_I64 = 27,
  GGML_TYPE_F64 = 28,
  GGML_TYPE_IQ1_M = 29,
  GGML_TYPE_COUNT,
};

enum gguf_metadata_value_type : uint32_t
{
  // The value is a 8-bit unsigned integer.
  GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
  // The value is a 8-bit signed integer.
  GGUF_METADATA_VALUE_TYPE_INT8 = 1,
  // The value is a 16-bit unsigned little-endian integer.
  GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
  // The value is a 16-bit signed little-endian integer.
  GGUF_METADATA_VALUE_TYPE_INT16 = 3,
  // The value is a 32-bit unsigned little-endian integer.
  GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
  // The value is a 32-bit signed little-endian integer.
  GGUF_METADATA_VALUE_TYPE_INT32 = 5,
  // The value is a 32-bit IEEE754 floating point number.
  GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
  // The value is a boolean.
  // 1-byte value where 0 is false and 1 is true.
  // Anything else is invalid, and should be treated as either the model being
  // invalid or the reader being buggy.
  GGUF_METADATA_VALUE_TYPE_BOOL = 7,
  // The value is a UTF-8 non-null-terminated string, with length prepended.
  GGUF_METADATA_VALUE_TYPE_STRING = 8,
  // The value is an array of other values, with the length and type prepended.
  ///
  // Arrays can be nested, and the length of the array is the number of
  // elements in the array, not the number of bytes.
  GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
  // The value is a 64-bit unsigned little-endian integer.
  GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
  // The value is a 64-bit signed little-endian integer.
  GGUF_METADATA_VALUE_TYPE_INT64 = 11,
  // The value is a 64-bit IEEE754 floating point number.
  GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

// A string in GGUF.
struct gguf_string_t
{
  // The length of the string, in bytes.
  uint64_t len;
  // The string as a UTF-8 non-null-terminated string.
  char string[0];
} __PACKED;

union gguf_metadata_elem_t
{
  uint8_t uint8;
  int8_t int8;
  uint16_t uint16;
  int16_t int16;
  uint32_t uint32;
  int32_t int32;
  float float32;
  uint64_t uint64;
  int64_t int64;
  double float64;
  bool bool_;
  gguf_string_t string;
} __PACKED;

union gguf_metadata_value_t
{
  gguf_metadata_elem_t value;
  struct
  {
    // Any value type is valid, including arrays.
    gguf_metadata_value_type type;
    // Number of elements, not bytes
    uint64_t len;
    // The array of values.
    uint8_t array[0];
  } __PACKED array;
} __PACKED;

struct gguf_metadata_kv_t
{
  // The key of the metadata. It is a standard GGUF string, with the following
  // caveats:
  // - It must be a valid ASCII string.
  // - It must be a hierarchical key, where each segment is `lower_snake_case`
  // and separated by a `.`.
  // - It must be at most 2^16-1/65535 bytes long.
  // Any keys that do not follow these rules are invalid.
  gguf_string_t key;

  // The type of the value.
  // Must be one of the `gguf_metadata_value_type` values.
  gguf_metadata_value_type value_type;
  gguf_metadata_value_t value;
} __PACKED;

struct gguf_header_t
{
  // Magic number to announce that this is a GGUF file.
  // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
  // Your executor might do little-endian byte order, so it might be
  // check for 0x46554747 and letting the endianness cancel out.
  // Consider being *very* explicit about the byte order here.
  uint32_t magic;
  // The version of the format implemented.
  // Must be `3` for version described in this spec, which introduces
  // big-endian support.
  //
  // This version should only be increased for structural changes to the
  // format. Changes that do not affect the structure of the file should
  // instead update the metadata to signify the change.
  uint32_t version;
  // The number of tensors in the file.
  // This is explicit, instead of being included in the metadata, to ensure it
  // is always present for loading the tensors.
  uint64_t tensor_count;
  // The number of metadata key-value pairs.
  uint64_t metadata_kv_count;
  // The metadata key-value pairs.
  gguf_metadata_kv_t metadata_kv[0];
} __PACKED;

inline uint64_t
align_offset (uint64_t offset, uint64_t align)
{
  return offset + (align - (offset % align)) % align;
}

struct gguf_tensor_info_t
{
  // The name of the tensor. It is a standard GGUF string, with the caveat that
  // it must be at most 64 bytes long.
  // gguf_string_t name;
  // The number of dimensions in the tensor.
  // Currently at most 4, but this may change in the future.
  uint32_t n_dimensions;
  // The dimensions of the tensor.
  union
  {
    struct
    {
      uint64_t dimensions[4];
      ggml_type type;
      uint64_t offset;
    } __PACKED dims4;
    struct
    {
      uint64_t dimensions[3];
      ggml_type type;
      uint64_t offset;
    } __PACKED dims3;
    struct
    {
      uint64_t dimensions[2];
      ggml_type type;
      uint64_t offset;
    } __PACKED dims2;
    struct
    {
      uint64_t dimensions[1];
      ggml_type type;
      uint64_t offset;
    } __PACKED dims1;
  } __PACKED;
  // The type of the tensor.
  // The offset of the tensor's data in this file in bytes.
  //
  // This offset is relative to `tensor_data`, not to the start
  // of the file, to make it easier for writers to write the file.
  // Readers should consider exposing this offset relative to the
  // file to make it easier to read the data.
  //
  // Must be a multiple of `ALIGNMENT`. That is, `align_offset(offset) ==
  // offset`.
} __PACKED;

struct gguf_file_t
{
  // The header of the file.
  gguf_header_t header;

  // Tensor infos, which can be used to locate the tensor data.
  gguf_tensor_info_t tensor_infos[0];

  // Padding to the nearest multiple of `ALIGNMENT`.
  //
  // That is, if `sizeof(header) + sizeof(tensor_infos)` is not a multiple of
  // `ALIGNMENT`, this padding is added to make it so.
  //
  // This can be calculated as `align_offset(position) - position`, where
  // `position` is the position of the end of `tensor_infos` (i.e.
  // `sizeof(header) + sizeof(tensor_infos)`).
  // uint8_t _padding[];

  // Tensor data.
  //
  // This is arbitrary binary data corresponding to the weights of the model.
  // This data should be close or identical to the data in the original model
  // file, but may be different due to quantization or other optimizations for
  // inference. Any such deviations should be recorded in the metadata or as
  // part of the architecture definition.
  //
  // Each tensor's data must be stored within this array, and located through
  // its `tensor_infos` entry. The offset of each tensor's data must be a
  // multiple of `ALIGNMENT`, and the space between tensors should be padded to
  // `ALIGNMENT` bytes.
  // uint8_t tensor_data[];
} __PACKED;

extern gguf_header_t *map_gguf_file (int fd);

inline gguf_header_t *
map_gguf_file (int fd)
{
  struct stat stat;
  auto ret = ::fstat (fd, &stat);
  if (ret < 0)
    {
      return nullptr;
    }

  auto *mapped_data
      = ::mmap (nullptr, stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped_data == MAP_FAILED)
    {
      return nullptr;
    }

  return reinterpret_cast<gguf_header_t *> (mapped_data);
}

template <typename T> struct __is_vector
{
  static constexpr bool value = false;
};

template <typename T, typename _Alloc>
struct __is_vector<std::vector<T, _Alloc> >
{
  static constexpr bool value = true;
};

template <typename T>
constexpr gguf_metadata_value_type
__to_gguf_meta_value_type ()
{
  static_assert (
      std::is_same<T, uint8_t>::value || std::is_same<T, uint16_t>::value
          || std::is_same<T, uint32_t>::value
          || std::is_same<T, uint64_t>::value || std::is_same<T, int8_t>::value
          || std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value
          || std::is_same<T, int64_t>::value || std::is_same<T, bool>::value
          || std::is_same<T, std::string>::value
          || std::is_same<T, float>::value || std::is_same<T, double>::value
          || __is_vector<T>::value,
      "unsupported type");

  if constexpr (std::is_same<T, int8_t>::value)
    {
      return GGUF_METADATA_VALUE_TYPE_INT8;
    }

  if constexpr (std::is_same<T, int16_t>::value)
    {
      return GGUF_METADATA_VALUE_TYPE_INT16;
    }

  if constexpr (std::is_same<T, int32_t>::value)
    {
      return GGUF_METADATA_VALUE_TYPE_INT32;
    }

  if constexpr (std::is_same<T, int64_t>::value)
    {
      return GGUF_METADATA_VALUE_TYPE_INT64;
    }

  if constexpr (std::is_same<T, uint8_t>::value)
    {
      return GGUF_METADATA_VALUE_TYPE_UINT8;
    }

  if constexpr (std::is_same<T, uint16_t>::value)
    {
      return GGUF_METADATA_VALUE_TYPE_UINT16;
    }

  if constexpr (std::is_same<T, uint32_t>::value)
    {
      return GGUF_METADATA_VALUE_TYPE_UINT32;
    }

  if constexpr (std::is_same<T, uint64_t>::value)
    {
      return GGUF_METADATA_VALUE_TYPE_UINT64;
    }

  if constexpr (std::is_same<T, float>::value)
    {
      return GGUF_METADATA_VALUE_TYPE_FLOAT32;
    }

  if constexpr (std::is_same<T, double>::value)
    {
      return GGUF_METADATA_VALUE_TYPE_FLOAT64;
    }

  if constexpr (std::is_same<T, bool>::value)
    {
      return GGUF_METADATA_VALUE_TYPE_BOOL;
    }

  if constexpr (std::is_same<T, std::string>::value)
    {
      return GGUF_METADATA_VALUE_TYPE_STRING;
    }

  if constexpr (__is_vector<T>::value)
    {
      return GGUF_METADATA_VALUE_TYPE_ARRAY;
    }
}

class GGUF
{
public:
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

  const char *__gguf_type_name[13]
      = { "GGUF_METADATA_VALUE_TYPE_UINT8",   "GGUF_METADATA_VALUE_TYPE_INT8",
          "GGUF_METADATA_VALUE_TYPE_UINT16",  "GGUF_METADATA_VALUE_TYPE_INT16",
          "GGUF_METADATA_VALUE_TYPE_UINT32",  "GGUF_METADATA_VALUE_TYPE_INT32",
          "GGUF_METADATA_VALUE_TYPE_FLOAT32", "GGUF_METADATA_VALUE_TYPE_BOOL",
          "GGUF_METADATA_VALUE_TYPE_STRING",  "GGUF_METADATA_VALUE_TYPE_ARRAY",
          "GGUF_METADATA_VALUE_TYPE_UINT64",  "GGUF_METADATA_VALUE_TYPE_INT64",
          "GGUF_METADATA_VALUE_TYPE_FLOAT64" };

  const char *__ggml_type_name[31] = {
    "GGML_TYPE_F32",     "GGML_TYPE_F16",     "GGML_TYPE_Q4_0",
    "GGML_TYPE_Q4_1",    "GGML_TYPE_Q4_2",    "GGML_TYPE_Q4_3",
    "GGML_TYPE_Q5_0",    "GGML_TYPE_Q5_1",    "GGML_TYPE_Q8_0",
    "GGML_TYPE_Q8_1",    "GGML_TYPE_Q2_K",    "GGML_TYPE_Q3_K",
    "GGML_TYPE_Q4_K",    "GGML_TYPE_Q5_K",    "GGML_TYPE_Q6_K",
    "GGML_TYPE_Q8_K",    "GGML_TYPE_IQ2_XXS", "GGML_TYPE_IQ2_XS",
    "GGML_TYPE_IQ3_XXS", "GGML_TYPE_IQ1_S",   "GGML_TYPE_IQ4_NL",
    "GGML_TYPE_IQ3_S",   "GGML_TYPE_IQ2_S",   "GGML_TYPE_IQ4_XS",
    "GGML_TYPE_I8",      "GGML_TYPE_I16",     "GGML_TYPE_I32",
    "GGML_TYPE_I64",     "GGML_TYPE_F64",     "GGML_TYPE_IQ1_M",
    "GGML_TYPE_COUNT",
  };

  struct __gguf_value_view
  {
    gguf_metadata_value_type value_type;
    gguf_metadata_value_t value;
  } __PACKED;

  struct __gguf_tensor_info_view
  {
    gguf_tensor_info_t *info;
    uint32_t ndim;
    uint64_t *dims;
    ggml_type dtype;
  };

  GGUF (std::string const &path) : gguf_ (nullptr), path_ (path), fd_ (-1) {}
  ~GGUF ()
  {
    if (gguf_)
      {
        // FIXME: unmap here
      }

    if (fd_ > 0)
      {
        ::close (fd_);
      }
  }
  int
  init ()
  {
    fd_ = ::open (path_.c_str (), O_RDONLY);
    if (fd_ < 0)
      {
        return fd_;
      }

    gguf_ = map_gguf_file (fd_);

    if (!gguf_)
      {
        return -1;
      }
    if (gguf_->magic != 0x46554747)
      {
        return -1;
      }

    fprintf (stderr,
             "magic: 0x%x\nversion: %u\ntensor_count: %lu\nmetadata_kv_count: "
             "%lu\n",
             gguf_->magic, gguf_->version, gguf_->tensor_count,
             gguf_->metadata_kv_count);

    struct gguf_metadata_kv_t *metadata = gguf_->metadata_kv;
    for (decltype (gguf_->metadata_kv_count) i = 0;
         i < gguf_->metadata_kv_count; ++i)
      {
        std::string key (metadata->key.string, metadata->key.len);
        auto *value
            = (__gguf_value_view *)(reinterpret_cast<uint8_t *> (metadata)
                                    + sizeof (gguf_string_t)
                                    + metadata->key.len);
        metadata_kv_[key] = value;

        auto value_type = value->value_type;
        if (value_type != GGUF_METADATA_VALUE_TYPE_ARRAY)
          {
            size_t offset = 0;
            if (value_type == GGUF_METADATA_VALUE_TYPE_STRING)
              {
                auto *string = &value->value.value.string;
                offset = sizeof (gguf_string_t) + string->len;
              }
            else
              {
                offset = __gguf_elem_size[value_type];
              }
            metadata
                = (struct gguf_metadata_kv_t *)((uint8_t *)value
                                                + sizeof (
                                                    gguf_metadata_value_type)
                                                + offset);
          }
        else
          {
            auto *array = &value->value.array;
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

            metadata
                = (struct gguf_metadata_kv_t *)((uint8_t *)value
                                                + sizeof (
                                                    gguf_metadata_value_type)
                                                + sizeof (*array) + offset);
          }
      }

    // parse tensor infos
    auto *tensor_name = (struct gguf_string_t *)metadata;
    for (decltype (gguf_->tensor_count) i = 0; i < gguf_->tensor_count; ++i)
      {
        std::string name (tensor_name->string, tensor_name->len);
        auto *tensor_info
            = (struct gguf_tensor_info_t *)((uint8_t *)tensor_name
                                            + sizeof (gguf_string_t)
                                            + tensor_name->len);

        uint32_t ndim = tensor_info->n_dimensions;
        uint64_t *dims = nullptr;
        ggml_type dtype = GGML_TYPE_F32;

        if (tensor_info->n_dimensions == 1)
          {
            dims = tensor_info->dims1.dimensions;
            dtype = tensor_info->dims1.type;
            tensor_name
                = (gguf_string_t *)((uint8_t *)tensor_info
                                    + sizeof (tensor_info->dims1)
                                    + sizeof (tensor_info->n_dimensions));
          }
        else if (tensor_info->n_dimensions == 2)
          {
            dims = tensor_info->dims2.dimensions;
            dtype = tensor_info->dims2.type;
            tensor_name
                = (gguf_string_t *)((uint8_t *)tensor_info
                                    + sizeof (tensor_info->dims2)
                                    + sizeof (tensor_info->n_dimensions));
          }
        else if (tensor_info->n_dimensions == 3)
          {
            dims = tensor_info->dims3.dimensions;
            dtype = tensor_info->dims3.type;
            tensor_name
                = (gguf_string_t *)((uint8_t *)tensor_info
                                    + sizeof (tensor_info->dims3)
                                    + sizeof (tensor_info->n_dimensions));
          }
        else if (tensor_info->n_dimensions == 4)
          {
            dims = tensor_info->dims3.dimensions;
            dtype = tensor_info->dims3.type;
            tensor_name
                = (gguf_string_t *)((uint8_t *)tensor_info
                                    + sizeof (tensor_info->dims4)
                                    + sizeof (tensor_info->n_dimensions));
          }

        tensor_infos_[name] = { tensor_info, ndim, dims, dtype };
      }

    return 0;
  }

  const __gguf_tensor_info_view *
  get_tensor_info (std::string const &name)
  {
    auto pos = tensor_infos_.find (name);
    if (pos != tensor_infos_.cend ())
      {
        return &pos->second;
      }
    return nullptr;
  }

  const auto &
  get_all_tensor_infos ()
  {
    return tensor_infos_;
  }

  template <typename T>
  int
  get (std::string const &key, T &result)
  {
    auto pos = metadata_kv_.find (key);
    if (pos == metadata_kv_.cend ())
      {
        return -1;
      }

    auto *kv = pos->second;
    if (__to_gguf_meta_value_type<T> () != kv->value_type)
      {
        return -1;
      }

    if constexpr (__is_vector<T>::value)
      {
        auto *array = &kv->value.array;
        if (array->type
            != __to_gguf_meta_value_type<typename T::value_type> ())
          {
            return -1;
          }

        gguf_metadata_elem_t *elem = (gguf_metadata_elem_t *)array->array;
        for (decltype (array->len) i = 0; i < array->len; ++i)
          {
            typename T::value_type v;
            auto size = parse_value_ (elem, v);
            result.push_back (v);
            elem = (gguf_metadata_elem_t *)((uint8_t *)elem + size);
          }

        return 0;
      }

    parse_value_<T> (&kv->value.value, result);
    return 0;
  }

  std::vector<std::pair<std::string, gguf_metadata_value_type> >
  get_all_metadata_keys ()
  {
    std::vector<std::pair<std::string, gguf_metadata_value_type> > keys;

    for (auto &kv : metadata_kv_)
      {
        auto value_type = kv.second->value_type;
        keys.push_back ({ kv.first, value_type });
      }
    return keys;
  }

private:
  gguf_header_t *gguf_;
  std::string const path_;
  int fd_;

  std::unordered_map<std::string, __gguf_value_view *> metadata_kv_;
  std::unordered_map<std::string, __gguf_tensor_info_view> tensor_infos_;
  template <typename T>
  size_t
  parse_value_ (gguf_metadata_elem_t *value, T &result)
  {
    if constexpr (__to_gguf_meta_value_type<T> ()
                  == GGUF_METADATA_VALUE_TYPE_INT8)
      {
        result = value->int8;
        return __gguf_elem_size[GGUF_METADATA_VALUE_TYPE_INT8];
      }

    if constexpr (__to_gguf_meta_value_type<T> ()
                  == GGUF_METADATA_VALUE_TYPE_INT16)
      {
        result = value->int16;
        return __gguf_elem_size[GGUF_METADATA_VALUE_TYPE_INT16];
      }

    if constexpr (__to_gguf_meta_value_type<T> ()
                  == GGUF_METADATA_VALUE_TYPE_INT32)
      {
        result = value->int32;
        return __gguf_elem_size[GGUF_METADATA_VALUE_TYPE_INT32];
      }

    if constexpr (__to_gguf_meta_value_type<T> ()
                  == GGUF_METADATA_VALUE_TYPE_INT64)
      {
        result = value->int64;
        return __gguf_elem_size[GGUF_METADATA_VALUE_TYPE_INT64];
      }

    if constexpr (__to_gguf_meta_value_type<T> ()
                  == GGUF_METADATA_VALUE_TYPE_UINT8)
      {
        result = value->uint8;
        return __gguf_elem_size[GGUF_METADATA_VALUE_TYPE_UINT8];
      }

    if constexpr (__to_gguf_meta_value_type<T> ()
                  == GGUF_METADATA_VALUE_TYPE_UINT16)
      {
        result = value->uint16;
        return __gguf_elem_size[GGUF_METADATA_VALUE_TYPE_UINT16];
      }

    if constexpr (__to_gguf_meta_value_type<T> ()
                  == GGUF_METADATA_VALUE_TYPE_UINT32)
      {
        result = value->uint32;
        return __gguf_elem_size[GGUF_METADATA_VALUE_TYPE_UINT32];
      }

    if constexpr (__to_gguf_meta_value_type<T> ()
                  == GGUF_METADATA_VALUE_TYPE_UINT64)
      {
        result = value->uint64;
        return __gguf_elem_size[GGUF_METADATA_VALUE_TYPE_UINT64];
      }

    if constexpr (__to_gguf_meta_value_type<T> ()
                  == GGUF_METADATA_VALUE_TYPE_UINT64)
      {
        result = value->uint8;
        return __gguf_elem_size[GGUF_METADATA_VALUE_TYPE_UINT64];
      }

    if constexpr (__to_gguf_meta_value_type<T> ()
                  == GGUF_METADATA_VALUE_TYPE_BOOL)
      {
        result = value->bool_;
        return __gguf_elem_size[GGUF_METADATA_VALUE_TYPE_BOOL];
      }

    if constexpr (__to_gguf_meta_value_type<T> ()
                  == GGUF_METADATA_VALUE_TYPE_FLOAT32)
      {
        result = value->float32;
        return __gguf_elem_size[GGUF_METADATA_VALUE_TYPE_FLOAT32];
      }

    if constexpr (__to_gguf_meta_value_type<T> ()
                  == GGUF_METADATA_VALUE_TYPE_FLOAT64)
      {
        result = value->float64;
        return __gguf_elem_size[GGUF_METADATA_VALUE_TYPE_FLOAT64];
      }

    if constexpr (__to_gguf_meta_value_type<T> ()
                  == GGUF_METADATA_VALUE_TYPE_STRING)
      {
        result = std::string (value->string.string, value->string.len);
        return sizeof (value->string) + value->string.len;
      }
  }
};
#endif
