#ifndef __VKLLAMA_SHADER_CONSTANTS_H__
#define __VKLLAMA_SHADER_CONSTANTS_H__

#include <initializer_list>
#include <iterator>
#include <tuple>
#include <utility>
#include <vector>

namespace vkllama
{
class ShaderConstants
{
public:
  ShaderConstants () = default;

  template <typename... Args> ShaderConstants (Args &&...args)
  {
    (push_back (args), ...);
  }

  template <typename T>
  ShaderConstants (std::initializer_list<T> const &values)
  {
    for (auto const &v : values)
      {
        push_back (v);
      }
  }

  template <typename T>
  void
  push_back (T const v)
  {
    const auto offset = data_.size ();
    const uint8_t *p = reinterpret_cast<const uint8_t *> (&v);
    const auto n = sizeof (T);
    for (size_t i = 0; i < n; ++i)
      {
        data_.push_back (p[i]);
      }

    offset_.push_back (offset);
    sizes_.push_back (sizeof (T));
  }

  size_t
  elem_num () const
  {
    return offset_.size ();
  }

  size_t
  bytes () const
  {
    return data_.size ();
  }

  const uint8_t *
  data () const
  {
    return data_.data ();
  }

  std::vector<size_t> const &
  offsets () const
  {
    return offset_;
  }

  std::vector<size_t> const &
  sizes () const
  {
    return sizes_;
  }

private:
  std::vector<uint8_t> data_;
  std::vector<size_t> offset_;
  std::vector<size_t> sizes_;
};

}

#endif
