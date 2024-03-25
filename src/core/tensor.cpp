#include "tensor.h"
#include "allocator.h"
#include "gpu_device.h"
#include <atomic>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

VkTensor::VkTensor()
  : c_(0)
  , h_(0)
  , w_(0)
  , bytes_(sizeof(float))
  , dev_(nullptr)
  , visable_(false)
  , data_(VK_NULL_HANDLE)
  , status_(nullptr)
{
    mem_ = { VK_NULL_HANDLE, 0, 0, nullptr, 0 };
}
VkTensor::VkTensor(const int c,
                   const int h,
                   const int w,
                   GPUDevice* dev,
                   const bool visable)
  : c_(c)
  , h_(h)
  , w_(w)
  , dev_(dev)
  , visable_(visable)
  , data_(VK_NULL_HANDLE)
  , status_(nullptr)
{
    bytes_ = sizeof(float) * c * h * w;
    bytes_ = (bytes_ + 63) / 64 * 64;
    mem_ = { VK_NULL_HANDLE, 0, 0, nullptr, 0 };
}

VkTensor::VkTensor(const VkTensor& rhs)
  : c_(rhs.channels())
  , h_(rhs.height())
  , w_(rhs.width())
  , bytes_(rhs.bytes_)
  , dev_(rhs.dev_)
  , visable_(rhs.visable())
  , data_(rhs.data_)
  , mem_(rhs.mem_)
  , status_(rhs.status_)
{
    if (status_) {
        status_->ref_.fetch_add(1);
    }
}

VkTensor::VkTensor(VkTensor&& rhs)
  : c_(rhs.channels())
  , h_(rhs.height())
  , w_(rhs.width())
  , bytes_(rhs.bytes())
  , dev_(rhs.dev_)
  , visable_(rhs.visable_)
  , data_(rhs.data_)
  , mem_(rhs.mem_)
  , status_(rhs.status_)
{
    rhs.status_ = nullptr;
}

VkTensor&
VkTensor::operator=(VkTensor const& rhs)
{
    c_ = rhs.channels();
    h_ = rhs.height();
    w_ = rhs.width();
    bytes_ = rhs.bytes_;
    dev_ = rhs.dev_;
    visable_ = rhs.visable_;
    data_ = rhs.data_;
    mem_ = rhs.mem_;
    status_ = rhs.status_;

    if (status_) {
        status_->ref_.fetch_add(1);
    }
    return *this;
}

VkResult
VkTensor::create()
{
    {
        VkBufferCreateInfo createInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                          nullptr,
                                          0,
                                          bytes_,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                          VK_SHARING_MODE_EXCLUSIVE,
                                          0,
                                          nullptr };

        auto ret = vkCreateBuffer(dev_->device(), &createInfo, nullptr, &data_);
        if (ret != VK_SUCCESS)
            return ret;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev_->device(), data_, &req);

    auto ret = dev_->allocator().allocate(req, &mem_, visable_);
    if (ret != VK_SUCCESS) {
        vkDestroyBuffer(dev_->device(), data_, nullptr);
        data_ = VK_NULL_HANDLE;
        return ret;
    }

    ret = vkBindBufferMemory(dev_->device(), data_, mem_.mem, mem_.offset);
    if (ret != VK_SUCCESS) {
        vkDestroyBuffer(dev_->device(), data_, nullptr);
        dev_->allocator().free(mem_);
        data_ = VK_NULL_HANDLE;
        return ret;
    }

    status_ = new __TensorStatus();
    status_->access_flags_.store(0);
    status_->pipeline_stage_.store(0);
    status_->ref_.store(1);
    return VK_SUCCESS;
}

size_t
VkTensor::bytes() const
{
    return bytes_;
}

void*
VkTensor::host()
{
    return mem_.host;
}

VkAccessFlags
VkTensor::access_flags() const
{
    return status_ ? status_->access_flags_.load() : 0;
}

VkPipelineStageFlags
VkTensor::pipeline_stage() const
{
    return status_ ? status_->pipeline_stage_.load() : 0;
}

void
VkTensor::set_access_flags(VkAccessFlags flags)
{
    if (!status_)
        return;
    status_->access_flags_.store(flags);
}

void
VkTensor::set_pipeline_stage(VkPipelineStageFlags stage)
{
    if (!status_)
        return;
    status_->pipeline_stage_.store(stage);
}

bool
VkTensor::visable() const
{
    return visable_;
}

size_t
VkTensor::channels() const
{
    return c_;
}

size_t
VkTensor::height() const
{
    return h_;
}

size_t
VkTensor::width() const
{
    return w_;
}

size_t VkTensor::size() const
{
	return c_ * h_ * w_;
}

VkBuffer&
VkTensor::data()
{
    return data_;
}

VkResult
VkTensor::flush()
{
    if (!visable_) {
        return VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS;
    }

    VkMappedMemoryRange range = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                                  nullptr,
                                  mem_.mem,
                                  mem_.offset,
                                  mem_.size };

    return vkFlushMappedMemoryRanges(dev_->device(), 1, &range);
}

VkResult
VkTensor::invalid()
{
    if (!visable_) {
        return VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS;
    }

    VkMappedMemoryRange range = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                                  nullptr,
                                  mem_.mem,
                                  mem_.offset,
                                  mem_.size };

    return vkInvalidateMappedMemoryRanges(dev_->device(), 1, &range);
}

VkTensor::~VkTensor()
{
    if (status_ && status_->ref_.fetch_sub(1) == 1) {
        if (data_ != VK_NULL_HANDLE) {
            dev_->allocator().free(mem_);
            vkDestroyBuffer(dev_->device(), data_, nullptr);
        }

        delete status_;
    }
}

VkTensor
VkTensor::like(const VkTensor& tensor)
{
    VkTensor tmp(tensor.channels(),
                 tensor.height(),
                 tensor.width(),
                 tensor.dev_,
                 tensor.visable());
    return tmp;
}
