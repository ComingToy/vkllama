#ifndef __VKLLAMA_CONTEXT_H__
#define __VKLLAMA_CONTEXT_H__

#include <any>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
namespace vkllama
{
class Context
{
public:
  Context () = default;

  template <typename T>
  void
  set (std::string const &key, T &&value)
  {
    data_[key] = std::forward (value);
  }

  template <typename T>
  bool
  get (std::string const &key, T &ref)
  {
    auto pos = data_.find (key);
    if (pos == data_.cend ())
      {
        return false;
      }

    try
      {
        ref = std::any_cast<T> (*pos);
      }
    catch (std::bad_any_cast const &e)
      {
        return false;
      }

    return true;
  };

  bool
  has (std::string const &key) const
  {
    return data_.count (key) > 0;
  }

private:
  std::map<std::string, std::any> data_;
};
}
#endif
