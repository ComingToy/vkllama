#ifndef __VKLLAMA_TENSOR__
#define __VKLLAMA_TENSOR__

#include "absl/status/status.h"
#include "gpu_device.h"
#include "src/core/float.h"
#include <atomic>
#include <set>
#include <type_traits>
#include <vulkan/vulkan.h>

namespace vkllama
{
class Tensor
{
public:
  typedef enum : int
  {
    FP32 = 0,
    FP16,
    UINT32,
    INT8,
    Q8_0, // block-wise quantize
  } DType;

  static Tensor like (Tensor const &);
  Tensor ();
  Tensor (const int c, const int h, const int w, GPUDevice *dev,
          DType const dtype = FP32, const bool visable = false);

  Tensor &operator= (Tensor const &);
  Tensor (const Tensor &rhs);
  Tensor (Tensor &&rhs);

  ~Tensor ();

  size_t channels () const;
  size_t height () const;
  size_t width () const;
  size_t size () const;

  absl::Status reshape (size_t const c, size_t const h, size_t const w);
  VkAccessFlags access_flags () const;
  VkPipelineStageFlags pipeline_stage () const;
  void set_access_flags (VkAccessFlags access_flags);
  void set_pipeline_stage (VkPipelineStageFlags stage_flags);

  VkBuffer &data ();

  absl::Status create ();
  size_t bytes () const;
  bool visable () const;
  DType dtype () const;
  size_t elem_bytes () const;
  absl::Status flush ();
  absl::Status invalid ();

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
