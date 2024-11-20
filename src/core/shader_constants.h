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
  ShaderConstants (ShaderConstants const &other)
      : data_ (other.data_), offset_ (other.offset_), sizes_ (other.sizes_)
  {
  }

  template <typename... Args> ShaderConstants (Args const &...args)
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

  ShaderConstants &
  operator+= (ShaderConstants const &rhs)
  {
    const auto *p = rhs.data ();
    const auto &sizes = rhs.sizes ();

    for (auto const size : sizes)
      {
        push_back (p, size);
        p += size;
      }
    return *this;
  }

  ShaderConstants &
  operator= (ShaderConstants const &rhs)
  {
    data_ = rhs.data_;
    offset_ = rhs.offset_;
    sizes_ = rhs.sizes_;
    return *this;
  }

  ShaderConstants
  operator+ (const ShaderConstants &rhs)
  {
    ShaderConstants constant (*this);
    constant += rhs;
    return constant;
  }

  template <typename T>
  void
  push_back (T const v)
  {
    push_back (reinterpret_cast<const uint8_t *> (&v), sizeof (T));
  }

  void
  push_back (uint8_t const *v, size_t len)
  {
    const auto offset = data_.size ();
    for (size_t i = 0; i < len; ++i)
      {
        data_.push_back (v[i]);
      }

    offset_.push_back (offset);
    sizes_.push_back (len);
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
