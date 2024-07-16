#ifndef __VKLLAMA_CPP_GPU_DEVICE_H__
#define __VKLLAMA_CPP_GPU_DEVICE_H__
#include "vk_mem_alloc.h"
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

#define VMA_IMPLEMENTATION

namespace vkllama
{
class GPUDevice
{
public:
  GPUDevice (int dev = 0);
  VmaAllocator &allocator ();
  VkDevice &device ();
  VkPhysicalDevice &phy ();
  VkInstance &instance ();

  uint32_t find_mem (uint32_t typeBits,
                     VkMemoryPropertyFlags properties) const;
  uint32_t require_queue (VkQueueFlags flags) const;
  const VkPhysicalDeviceLimits &limits () const;
  const float timestamp_period () const;
  VkResult init ();
  bool support_descriptor_templ_update () const;
  bool support_16bit_storage () const;
  bool support_fp16_arithmetic () const;
  bool support_int8_arithmetic () const;
  bool support_pipeline_statistics () const;

  ~GPUDevice ();

private:
  VkResult create_instance_ ();
  VkResult init_device_ ();
  VkInstance instance_;
  VkPhysicalDevice physicalDev_;
  VkDevice device_;
  std::vector<VkExtensionProperties> instanceExts_;
  VkPhysicalDeviceFeatures physicalFeats_;
  VkPhysicalDeviceMemoryProperties physicalDevMemProperties_;
  std::vector<VkExtensionProperties> physicalDevExts_;
  std::vector<VkQueueFamilyProperties> physicalDevQueueProperties_;
  VkPhysicalDeviceProperties physicalDevProperties_;
  const int dev_;
  VmaAllocator allocator_;
  uint32_t version_;
  bool support_descriptor_templ_update_;
  bool support_16bit_storage_;
  bool support_shader_fp16_arithmetic_;
  bool support_shader_int8_arithmetic_;
};
}

#endif
