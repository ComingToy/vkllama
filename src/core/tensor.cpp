#include "tensor.h"
#include "gpu_device.h"
#include "vk_mem_alloc.h"
#include <atomic>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

VkTensor::VkTensor ()
    : c_ (0), h_ (0), w_ (0), dev_ (nullptr), visable_ (false), dtype_ (FP32),
      data_ (VK_NULL_HANDLE), status_ (nullptr)
{
  mem_ = { 0, 0, 0, 0, 0, 0, 0 };
}
VkTensor::VkTensor (const int c, const int h, const int w, GPUDevice *dev,
                    const DType dtype, const bool visable)
    : c_ (c), h_ (h), w_ (w), dev_ (dev), visable_ (visable), dtype_ (dtype),
      data_ (VK_NULL_HANDLE), status_ (nullptr)
{
  mem_ = { 0, 0, 0, 0, 0, 0, 0 };
}

VkTensor::VkTensor (const VkTensor &rhs)
    : c_ (rhs.channels ()), h_ (rhs.height ()), w_ (rhs.width ()),
      dev_ (rhs.dev_), visable_ (rhs.visable ()), dtype_ (rhs.dtype_),
      data_ (rhs.data_), mem_ (rhs.mem_), allocation_ (rhs.allocation_),
      status_ (rhs.status_)
{
  if (status_)
    {
      status_->ref_.fetch_add (1);
    }
}

VkTensor::VkTensor (VkTensor &&rhs)
    : c_ (rhs.channels ()), h_ (rhs.height ()), w_ (rhs.width ()),
      dev_ (rhs.dev_), visable_ (rhs.visable_), dtype_ (rhs.dtype_),
      data_ (rhs.data_), mem_ (rhs.mem_), allocation_ (rhs.allocation_),
      status_ (rhs.status_)
{
  rhs.status_ = nullptr;
}

VkTensor &
VkTensor::operator= (VkTensor const &rhs)
{

  if (rhs.status_)
    {
      rhs.status_->ref_.fetch_add (1);
    }

  release_ ();
  c_ = rhs.channels ();
  h_ = rhs.height ();
  w_ = rhs.width ();
  dev_ = rhs.dev_;
  visable_ = rhs.visable_;
  dtype_ = rhs.dtype_;
  data_ = rhs.data_;
  mem_ = rhs.mem_;
  allocation_ = rhs.allocation_;
  status_ = rhs.status_;

  return *this;
}

size_t
VkTensor::elem_bytes () const
{
  if (dtype_ == FP32)
    {
      return sizeof (float);
    }
  else if (dtype_ == UINT32)
    {
      return sizeof (uint32_t);
    }
  else if (dtype_ == FP16)
    {
      return sizeof (uint16_t);
    }
  else
    {
      return 0;
    }
}

VkResult
VkTensor::create ()
{
  const size_t align = dev_->limits ().nonCoherentAtomSize;
  auto bytes = elem_bytes () * w_ * h_ * c_;
  bytes = (bytes + align - 1) / align * align;
  {
    VkBufferCreateInfo createInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                      nullptr,
                                      0,
                                      bytes,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                          | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                      VK_SHARING_MODE_EXCLUSIVE,
                                      0,
                                      nullptr };

    VmaAllocationCreateFlags flags = 0;
    if (visable_)
      {
        flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
      }

    VmaAllocationCreateInfo allocInfo = {
      flags, VMA_MEMORY_USAGE_AUTO, 0, 0, 0, VK_NULL_HANDLE, nullptr, 0
    };

    auto ret = vmaCreateBuffer (dev_->allocator (), &createInfo, &allocInfo,
                                &data_, &allocation_, &mem_);
    if (ret != VK_SUCCESS)
      {
        return ret;
      }
  }

  status_ = new __TensorStatus ();
  status_->access_flags_.store (0);
  status_->pipeline_stage_.store (0);
  status_->ref_.store (1);
  return VK_SUCCESS;
}

size_t
VkTensor::bytes () const
{
  auto bytes = elem_bytes () * w_ * h_ * c_;
  const auto align = dev_->limits ().nonCoherentAtomSize;
  bytes = (bytes + align - 1) / align * align;
  return bytes;
}

void *
VkTensor::host ()
{
  return mem_.pMappedData;
}

VkAccessFlags
VkTensor::access_flags () const
{
  return status_ ? status_->access_flags_.load () : 0;
}

VkPipelineStageFlags
VkTensor::pipeline_stage () const
{
  return status_ ? status_->pipeline_stage_.load () : 0;
}

void
VkTensor::set_access_flags (VkAccessFlags flags)
{
  if (!status_)
    return;
  status_->access_flags_.store (flags);
}

void
VkTensor::set_pipeline_stage (VkPipelineStageFlags stage)
{
  if (!status_)
    return;
  status_->pipeline_stage_.store (stage);
}

bool
VkTensor::visable () const
{
  return visable_;
}

size_t
VkTensor::channels () const
{
  return c_;
}

size_t
VkTensor::height () const
{
  return h_;
}

size_t
VkTensor::width () const
{
  return w_;
}

size_t
VkTensor::size () const
{
  return c_ * h_ * w_;
}

VkBuffer &
VkTensor::data ()
{
  return data_;
}

VkResult
VkTensor::flush ()
{
  if (!visable_)
    {
      return VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS;
    }

  VkMappedMemoryRange range = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr,
                                mem_.deviceMemory, mem_.offset, mem_.size };

  return vkFlushMappedMemoryRanges (dev_->device (), 1, &range);
}

VkResult
VkTensor::invalid ()
{
  if (!visable_)
    {
      return VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS;
    }

  VkMappedMemoryRange range = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr,
                                mem_.deviceMemory, mem_.offset, mem_.size };

  return vkInvalidateMappedMemoryRanges (dev_->device (), 1, &range);
}

void
VkTensor::release_ ()
{
  if (status_ && status_->ref_.fetch_sub (1) == 1)
    {
      if (data_ != VK_NULL_HANDLE)
        {
          vmaDestroyBuffer (dev_->allocator (), data_, allocation_);
        }

      delete status_;
    }
}

VkTensor::DType
VkTensor::dtype () const
{
  return dtype_;
}

VkTensor::~VkTensor () { release_ (); }

VkTensor
VkTensor::like (const VkTensor &tensor)
{
  VkTensor tmp (tensor.channels (), tensor.height (), tensor.width (),
                tensor.dev_, tensor.dtype (), tensor.visable ());
  return tmp;
}
