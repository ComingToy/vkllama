#ifndef __VKLLAMA_TENSOR__
#define __VKLLAMA_TENSOR__

#include "gpu_device.h"
#include "src/core/float.h"
#include <atomic>
#include <set>
#include <type_traits>
#include <vulkan/vulkan.h>

namespace vkllama
{
class VkTensor
{
public:
  typedef enum : int
  {
    FP32 = 0,
    FP16,
    UINT32
  } DType;

  template <typename T> DType static to_dtype ()
  {
    if (std::is_same<T, float>::value)
      {
        return FP32;
      }
    else if (std::is_same<T, __vkllama_fp16_t>::value)
      {
        return FP16;
      }
    else
      {
        return UINT32;
      }
  }

  static VkTensor like (VkTensor const &);
  VkTensor ();
  VkTensor (const int c, const int h, const int w, GPUDevice *dev,
            DType const dtype = FP32, const bool visable = false);

  VkTensor &operator= (VkTensor const &);
  VkTensor (const VkTensor &rhs);
  VkTensor (VkTensor &&rhs);

  ~VkTensor ();

  size_t channels () const;
  size_t height () const;
  size_t width () const;
  size_t size () const;
  VkAccessFlags access_flags () const;
  VkPipelineStageFlags pipeline_stage () const;
  void set_access_flags (VkAccessFlags access_flags);
  void set_pipeline_stage (VkPipelineStageFlags stage_flags);

  VkBuffer &data ();

  VkResult create ();
  size_t bytes () const;
  bool visable () const;
  DType dtype () const;
  size_t elem_bytes () const;
  VkResult flush ();
  VkResult invalid ();

  void *host ();

private:
  int c_;
  int h_;
  int w_;

  GPUDevice *dev_;
  bool visable_;
  DType dtype_;
  VkBuffer data_;
  struct __TensorStatus
  {
    std::atomic<VkAccessFlags> access_flags_;
    std::atomic<VkPipelineStageFlags> pipeline_stage_;
    std::atomic<int> ref_;
  };

  VmaAllocationInfo mem_;
  VmaAllocation allocation_;
  __TensorStatus *status_;
  void release_ ();
};

}

#endif
