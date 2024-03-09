#ifndef __VKLLAMA_CPP_GPU_DEVICE_H__
#define __VKLLAMA_CPP_GPU_DEVICE_H__
#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>

class Allocator;
class GPUDevice
{
  public:
    GPUDevice(int dev = 0);
    Allocator& allocator();
    VkDevice& device();
    uint32_t find_mem(uint32_t typeBits,
                      VkMemoryPropertyFlags properties) const;
    uint32_t require_queue(VkQueueFlags flags) const;
    const VkPhysicalDeviceLimits& limits() const;
    VkResult init();
    ~GPUDevice();

  private:
    VkResult create_instance_();
    VkResult init_device_();
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
    Allocator* allocator_;
};
#endif
