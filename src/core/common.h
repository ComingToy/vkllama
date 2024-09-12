#ifndef __VKLLAMA_COMMON_H__
#define __VKLLAMA_COMMON_H__
namespace vkllama
{
#define VKLLAMA_STATUS_OK(__s)                                                \
  if (!__s.ok ())                                                             \
    {                                                                         \
      return __s;                                                             \
    }
}
#endif
