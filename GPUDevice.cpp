#include "GPUDevice.h"
#include <algorithm>
#include <vector>
#include <vulkan/vulkan_core.h>

GPUDevice::GPUDevice(int dev)
  : dev_(dev)
{
    create_instance_();
    init_device_();
}

VkDevice&
GPUDevice::device()
{
    return device_;
}

VkResult
GPUDevice::create_instance_()
{
    static VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                         nullptr,
                                         "vkllama.cpp",
                                         1,
                                         "vkllama.cpp",
                                         1,
                                         VK_MAKE_VERSION(1, 2, 189) };
    const char* enabledLayers[] = {
#ifndef NDEBUG
        "VK_LAYER_KHRONOS_validation"
#endif
    };

    const char* exts[] = {
#if __APPLE__
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
#endif
    };

    VkInstanceCreateInfo instanceCreateInfo = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        nullptr,
#if __APPLE__
        VK_KHR_portability_enumeration,
#else
        0,
#endif
        &appInfo,
        1,
        enabledLayers,
        sizeof(exts) / sizeof(const char*),
        exts
    };

    auto ret = vkCreateInstance(&instanceCreateInfo, nullptr, &instance_);
    if (ret != VK_SUCCESS) {
        return ret;
    }

    uint32_t nExts = 0;
    ret = vkEnumerateInstanceExtensionProperties(nullptr, &nExts, nullptr);
    if (ret != VK_SUCCESS) {
        return ret;
    }
    ret = vkEnumerateInstanceExtensionProperties(
      nullptr, &nExts, instanceExts_.data());

    return ret;
}

VkResult
GPUDevice::init_device_()
{
    uint32_t ndev = 0;
    auto ret = vkEnumeratePhysicalDevices(instance_, &ndev, nullptr);
    if (ret != VK_SUCCESS) {
        return ret;
    }

    std::vector<VkPhysicalDevice> vkdev;
    ret = vkEnumeratePhysicalDevices(instance_, &ndev, vkdev.data());
    if (ret != VK_SUCCESS) {
        return ret;
    }

    if (vkdev.size() < dev_) {
        return VK_ERROR_DEVICE_LOST;
    }

    physicalDev_ = vkdev[dev_];
    {
        // get physical device features
        vkGetPhysicalDeviceFeatures(physicalDev_, &physicalFeats_);
        // get physical device extensions
        uint32_t n = 0;
        auto ret = vkEnumerateDeviceExtensionProperties(
          physicalDev_, nullptr, &n, nullptr);
        if (ret != VK_SUCCESS)
            return ret;
        physicalDevExts_.resize(n);
        vkEnumerateDeviceExtensionProperties(
          physicalDev_, nullptr, &n, physicalDevExts_.data());
        // get physical memory properties
        vkGetPhysicalDeviceMemoryProperties(physicalDev_,
                                            &physicalDevMemProperties_);
        // get physical queue properties
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDev_, &n, nullptr);
        physicalDevQueueProperties_.resize(n);
        vkGetPhysicalDeviceQueueFamilyProperties(
          physicalDev_, &n, physicalDevQueueProperties_.data());
    }

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

    {
        std::vector<std::vector<float>> priorities;
        for (auto i = 0; i < physicalDevQueueProperties_.size(); ++i) {
            const auto& feat = physicalDevQueueProperties_[i];
            priorities.emplace_back(feat.queueCount, 1.0f / feat.queueCount);
            VkDeviceQueueCreateInfo createInfo = {
                VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                nullptr,
                0,
                (uint32_t)i,
                feat.queueCount,
                priorities.back().data()
            };
            queueCreateInfos.push_back(createInfo);
        }
    }

    VkDeviceCreateInfo devCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                         nullptr,
                                         0,
                                         (uint32_t)queueCreateInfos.size(),
                                         queueCreateInfos.data(),
                                         0,
                                         nullptr,
                                         0,
                                         nullptr,
                                         &physicalFeats_ };

    return vkCreateDevice(physicalDev_, &devCreateInfo, nullptr, &device_);
}
