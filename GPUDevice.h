#ifndef __VKLLAMA_CPP_GPU_DEVICE_H__
#define __VKLLAMA_CPP_GPU_DEVICE_H__
#include <vector>
#include <vulkan/vulkan.hpp>

class GPUDevice
{
  public:
    GPUDevice(int dev = 0);
    VkDevice& device();

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
    const int dev_;
};
#endif
