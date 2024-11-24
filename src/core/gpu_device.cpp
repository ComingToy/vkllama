#include "gpu_device.h"
#include "absl/strings/str_format.h"
#include <algorithm>
#include <cstdio>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace vkllama
{
GPUDevice::GPUDevice (int dev)
    : physicalDev_ (VK_NULL_HANDLE), device_ (VK_NULL_HANDLE), dev_ (dev),
      version_ (0), support_descriptor_templ_update_ (false),
      support_16bit_storage_ (false), support_8bit_storage_ (false),
      support_shader_fp16_arithmetic_ (false),
      support_shader_int8_arithmetic_ (false)
{
}

absl::Status
GPUDevice::init ()
{
  auto ret = create_instance_ ();
  if (!ret.ok ())
    return ret;

  ret = init_device_ ();
  if (!ret.ok ())
    return ret;

  VmaAllocatorCreateInfo createInfo
      = { 0,       physicalDev_, device_,   0,        nullptr, nullptr,
          nullptr, nullptr,      instance_, version_, nullptr };
  auto vkret = vmaCreateAllocator (&createInfo, &allocator_);
  if (vkret != VK_SUCCESS)
    {
      return absl::InternalError (
          absl::StrFormat ("failed at vmaCreateAllocator: %d", int (vkret)));
    }
  return absl::OkStatus ();
}

GPUDevice::~GPUDevice ()
{
  vmaDestroyAllocator (allocator_);
  vkDestroyDevice (device_, nullptr);
  vkDestroyInstance (instance_, nullptr);
}

VmaAllocator &
GPUDevice::allocator ()
{
  return allocator_;
}

VkDevice &
GPUDevice::device ()
{
  return device_;
}

VkPhysicalDevice &
GPUDevice::phy ()
{
  return physicalDev_;
}

VkInstance &
GPUDevice::instance ()
{
  return instance_;
}

uint32_t
GPUDevice::find_mem (uint32_t typeBits, VkMemoryPropertyFlags properties) const
{
  for (size_t i = 0; i < sizeof (typeBits) * 8; ++i)
    {
      auto const &memProperties
          = physicalDevMemProperties_.memoryTypes[i].propertyFlags;
      if ((typeBits & (1 << i))
          && ((memProperties & properties) == properties))
        {
          return i;
        }
    }

  return 0;
}

absl::Status
GPUDevice::create_instance_ ()
{

  auto ret = vkEnumerateInstanceVersion (&version_);
  if (ret != VK_SUCCESS)
    {
      return absl::InternalError (absl::StrFormat (
          "failed at enumerate vulkan instance: %d", int (ret)));
    }

  static VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                       nullptr,
                                       "vkllama.cpp",
                                       1,
                                       "vkllama.cpp",
                                       1,
                                       version_ };
  const char *enabledLayers[] = {
#ifdef __VKLLAMA_DEBUG__
    "VK_LAYER_KHRONOS_validation"
#endif
  };

  const char *exts[] = {
#if __APPLE__
    VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
#endif
  };

  VkInstanceCreateInfo instanceCreateInfo
      = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
          nullptr,
#if __APPLE__
          VK_KHR_portability_enumeration,
#else
          0,
#endif
          &appInfo,
          sizeof (enabledLayers) / sizeof (const char *),
          enabledLayers,
          sizeof (exts) / sizeof (const char *),
          exts };

  ret = vkCreateInstance (&instanceCreateInfo, nullptr, &instance_);
  if (ret != VK_SUCCESS)
    {
      return absl::InternalError (
          absl::StrFormat ("create vulkan instance failed: %d", int (ret)));
    }

  uint32_t nExts = 0;
  ret = vkEnumerateInstanceExtensionProperties (nullptr, &nExts, nullptr);
  if (ret != VK_SUCCESS)
    {
      return absl::InternalError (absl::StrFormat (
          "enumerate vulkan instance extention properties failed: %d",
          int (ret)));
    }

  instanceExts_.resize (nExts);
  ret = vkEnumerateInstanceExtensionProperties (nullptr, &nExts,
                                                instanceExts_.data ());
  if (ret != VK_SUCCESS)
    {
      return absl::InternalError (absl::StrFormat (
          "enumerate instance extention properties failed: %d", int (ret)));
    }
  return absl::OkStatus ();
}

absl::Status
GPUDevice::init_device_ ()
{
  uint32_t ndev = 0;
  auto ret = vkEnumeratePhysicalDevices (instance_, &ndev, nullptr);
  if (ret != VK_SUCCESS)
    {
      return absl::InternalError (absl::StrFormat (
          "enumerate physical devices failed: %d", int (ret)));
    }

  std::vector<VkPhysicalDevice> vkdev;
  vkdev.resize (ndev);
  ret = vkEnumeratePhysicalDevices (instance_, &ndev, vkdev.data ());
  if (ret != VK_SUCCESS)
    {
      return absl::InternalError (absl::StrFormat (
          "enumerate physical devices failed: %d", int (ret)));
    }

  if (vkdev.size () < (size_t)dev_)
    {
      return absl::InternalError (
          absl::StrFormat ("target device %d not found.", dev_));
    }

  physicalDev_ = vkdev[dev_];
  {
    // get physical device features
    vkGetPhysicalDeviceFeatures (physicalDev_, &physicalFeats_);
    // get physical device extensions
    uint32_t n = 0;
    auto ret = vkEnumerateDeviceExtensionProperties (physicalDev_, nullptr, &n,
                                                     nullptr);
    if (ret != VK_SUCCESS)
      return absl::InternalError (absl::StrFormat (
          "enumerate device extension properties failed: %d", int (ret)));

    physicalDevExts_.resize (n);
    vkEnumerateDeviceExtensionProperties (physicalDev_, nullptr, &n,
                                          physicalDevExts_.data ());
    // get physical memory properties
    vkGetPhysicalDeviceMemoryProperties (physicalDev_,
                                         &physicalDevMemProperties_);
    // get physical queue properties
    vkGetPhysicalDeviceQueueFamilyProperties (physicalDev_, &n, nullptr);
    physicalDevQueueProperties_.resize (n);
    vkGetPhysicalDeviceQueueFamilyProperties (
        physicalDev_, &n, physicalDevQueueProperties_.data ());

    // get physical device properties
    vkGetPhysicalDeviceProperties (physicalDev_, &physicalDevProperties_);
  }

  {
    subgroupProperties_.sType
        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroupProperties_.pNext = NULL;

    physicalDevProperties2_.sType
        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    physicalDevProperties2_.pNext = &subgroupProperties_;

    vkGetPhysicalDeviceProperties2 (physicalDev_, &physicalDevProperties2_);
  }

  fprintf (stderr, "device subgroupSize = %u\n",
           subgroupProperties_.subgroupSize);
  std::vector<std::vector<float> > priorities;
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

  {
    for (size_t i = 0; i < physicalDevQueueProperties_.size (); ++i)
      {
        const auto &feat = physicalDevQueueProperties_[i];
        priorities.emplace_back (feat.queueCount, 0.5);
        VkDeviceQueueCreateInfo createInfo
            = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                nullptr,
                0,
                (uint32_t)i,
                feat.queueCount,
                priorities.back ().data () };
        queueCreateInfos.push_back (createInfo);
      }
  }

  std::vector<const char *> devExts = {
#if __APPLE__
    "VK_KHR_portability_subset"
#endif
  };

  {
    std::set<std::string> supported_exts;

    std::transform (physicalDevExts_.cbegin (), physicalDevExts_.cend (),
                    std::inserter (supported_exts, supported_exts.end ()),
                    [] (auto const &feat) { return feat.extensionName; });

    if (supported_exts.count (VK_KHR_DESCRIPTOR_UPDATE_TEMPLATE_EXTENSION_NAME)
        > 0)
      {
        devExts.push_back (VK_KHR_DESCRIPTOR_UPDATE_TEMPLATE_EXTENSION_NAME);
        support_descriptor_templ_update_ = true;
      }

    if (supported_exts.count (VK_KHR_16BIT_STORAGE_EXTENSION_NAME) > 0)
      {
        devExts.push_back (VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
        support_16bit_storage_ = true;
      }

    if (supported_exts.count (VK_KHR_8BIT_STORAGE_EXTENSION_NAME) > 0)
      {
        devExts.push_back (VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
        support_8bit_storage_ = true;
      }

    if (supported_exts.count (VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) > 0)
      {
        devExts.push_back (VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
        support_shader_fp16_arithmetic_ = true;
        support_shader_int8_arithmetic_ = true;
      }
  }

  static VkPhysicalDeviceShaderFloat16Int8Features feat_fp16_int8
      = { .sType
          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
          .pNext = nullptr };

  static VkPhysicalDevice16BitStorageFeatures feat_16bit
      = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
          &feat_fp16_int8 };

  static VkPhysicalDevice8BitStorageFeatures feat_8bit_storage
      = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
          &feat_16bit };

  static VkPhysicalDeviceFeatures2 feats
      = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &feat_8bit_storage };
  vkGetPhysicalDeviceFeatures2 (physicalDev_, &feats);

  {
    if (support_16bit_storage_)
      {
        support_16bit_storage_ = feat_16bit.storageBuffer16BitAccess
                                 && feat_16bit.storagePushConstant16;
      }

    if (support_8bit_storage_)
      {
        support_8bit_storage_ = feat_8bit_storage.storageBuffer8BitAccess
                                && feat_8bit_storage.storagePushConstant8;
      }

    if (support_shader_fp16_arithmetic_)
      {
        support_shader_fp16_arithmetic_ = feat_fp16_int8.shaderFloat16;
      }

    if (support_shader_int8_arithmetic_)
      {
        support_shader_int8_arithmetic_ = feat_fp16_int8.shaderInt8;
      }
  }

  VkDeviceCreateInfo devCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                       &feat_8bit_storage,
                                       0,
                                       (uint32_t)queueCreateInfos.size (),
                                       queueCreateInfos.data (),
                                       0,
                                       nullptr,
                                       (uint32_t)devExts.size (),
                                       devExts.data (),
                                       &physicalFeats_ };

  ret = vkCreateDevice (physicalDev_, &devCreateInfo, nullptr, &device_);
  if (ret != VK_SUCCESS)
    {
      return absl::InternalError (
          absl::StrFormat ("create device failed: %d", int (ret)));
    }

  return absl::OkStatus ();
}

uint32_t
GPUDevice::require_queue (VkQueueFlags flags) const
{
  uint32_t type = 0;
  for (size_t i = 0; i < physicalDevQueueProperties_.size (); ++i)
    {
      auto const &property = physicalDevQueueProperties_[i];
      if ((property.queueFlags & flags) == flags)
        {
          type = i;
          break;
        }
    }

  return type;
}

bool
GPUDevice::support_descriptor_templ_update () const
{
  return support_descriptor_templ_update_;
}

bool
GPUDevice::support_16bit_storage () const
{
  return support_16bit_storage_;
}

bool
GPUDevice::support_8bit_storage () const
{
  return support_8bit_storage_;
}

bool
GPUDevice::support_fp16_arithmetic () const
{
  return support_shader_fp16_arithmetic_;
}
bool
GPUDevice::support_int8_arithmetic () const
{
  return support_shader_int8_arithmetic_;
}

bool
GPUDevice::support_pipeline_statistics () const
{
  return physicalFeats_.pipelineStatisticsQuery;
}

VkPhysicalDeviceLimits const &
GPUDevice::limits () const
{
  return physicalDevProperties_.limits;
}

const float
GPUDevice::timestamp_period () const
{
  return physicalDevProperties_.limits.timestampPeriod;
}

size_t
GPUDevice::subgroup_size () const
{
  return subgroupProperties_.subgroupSize;
}
}

