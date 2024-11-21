#include "tensor.h"
#include "absl/strings/str_format.h"
#include "gpu_device.h"
#include "src/core/quants.h"
#include "vk_mem_alloc.h"
#include <atomic>
#include <cstddef>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

namespace vkllama
{
Tensor::Tensor ()
    : c_ (0), h_ (0), w_ (0), cs_ (0), hs_ (0), ws_ (0), dev_ (nullptr),
      visable_ (false), dtype_ (FP32), data_ (VK_NULL_HANDLE),
      status_ (nullptr)
{
  mem_ = { 0, 0, 0, 0, 0, 0, 0 };
}

Tensor::Tensor (const int c, const int h, const int w, GPUDevice *dev,
                const DType dtype, const bool visable)
    : Tensor (c, h, w, 0, 0, 0, dev, dtype, visable)
{
}

Tensor::Tensor (const int c, const int h, const int w, const int cs,
                const int hs, const int ws, GPUDevice *dev, const DType dtype,
                const bool visable)
    : c_ (c), h_ (h), w_ (w), cs_ (cs), hs_ (hs), ws_ (ws), dev_ (dev),
      visable_ (visable), dtype_ (dtype), data_ (VK_NULL_HANDLE),
      status_ (nullptr)
{
  mem_ = { 0, 0, 0, 0, 0, 0, 0 };
}

Tensor::Tensor (const Tensor &rhs)
    : c_ (rhs.channels ()), h_ (rhs.height ()), w_ (rhs.width ()),
      cs_ (rhs.cs ()), hs_ (rhs.hs ()), ws_ (rhs.ws ()), dev_ (rhs.dev_),
      visable_ (rhs.visable ()), dtype_ (rhs.dtype_), data_ (rhs.data_),
      mem_ (rhs.mem_), allocation_ (rhs.allocation_), status_ (rhs.status_)
{
  if (status_)
    {
      status_->ref_.fetch_add (1);
    }
}

Tensor::Tensor (Tensor &&rhs)
    : c_ (rhs.channels ()), h_ (rhs.height ()), w_ (rhs.width ()),
      cs_ (rhs.cs ()), hs_ (rhs.hs ()), ws_ (rhs.ws ()), dev_ (rhs.dev_),
      visable_ (rhs.visable_), dtype_ (rhs.dtype_), data_ (rhs.data_),
      mem_ (rhs.mem_), allocation_ (rhs.allocation_), status_ (rhs.status_)
{
  rhs.status_ = nullptr;
}

Tensor &
Tensor::operator= (Tensor const &rhs)
{
  if (rhs.status_)
    {
      rhs.status_->ref_.fetch_add (1);
    }

  release_ ();
  c_ = rhs.channels ();
  h_ = rhs.height ();
  w_ = rhs.width ();
  cs_ = rhs.cs ();
  hs_ = rhs.hs ();
  ws_ = rhs.ws ();

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
Tensor::elem_bytes () const
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
  else if (dtype_ == INT8 || dtype_ == Q8_0)
    {
      return sizeof (int8_t);
    }
  else
    {
      return 0;
    }
}

absl::Status
Tensor::create ()
{
  update_strides_ ();
  auto bytes = this->bytes ();

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

    VmaAllocationCreateInfo allocInfo
        = { flags,   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            0,       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            0,       VK_NULL_HANDLE,
            nullptr, 0 };

    auto ret = vmaCreateBuffer (dev_->allocator (), &createInfo, &allocInfo,
                                &data_, &allocation_, &mem_);
    if (ret != VK_SUCCESS)
      {
        return absl::InternalError (
            absl::StrFormat ("failed at creating vma buffer: %d", int (ret)));
      }
  }

  status_ = new __TensorStatus ();
  status_->access_flags_.store (0);
  status_->pipeline_stage_.store (0);
  status_->ref_.store (1);
  return absl::OkStatus ();
}

size_t
Tensor::bytes () const
{
  const auto align = dev_->limits ().nonCoherentAtomSize;
  return (c_ * cs_ + align - 1) / align * align;
}

void *
Tensor::host ()
{
  return mem_.pMappedData;
}

VkAccessFlags
Tensor::access_flags () const
{
  return status_ ? status_->access_flags_.load () : 0;
}

VkPipelineStageFlags
Tensor::pipeline_stage () const
{
  return status_ ? status_->pipeline_stage_.load () : 0;
}

void
Tensor::set_access_flags (VkAccessFlags flags)
{
  if (!status_)
    return;
  status_->access_flags_.store (flags);
}

void
Tensor::set_pipeline_stage (VkPipelineStageFlags stage)
{
  if (!status_)
    return;
  status_->pipeline_stage_.store (stage);
}

bool
Tensor::visable () const
{
  return visable_;
}

size_t
Tensor::channels () const
{
  return c_;
}

size_t
Tensor::height () const
{
  return h_;
}

size_t
Tensor::width () const
{
  return w_;
}

size_t
Tensor::cs () const
{
  return cs_;
}

size_t
Tensor::hs () const
{
  return hs_;
}

size_t
Tensor::ws () const
{
  return ws_;
}

void
Tensor::update_strides_ ()
{
  const auto dtype_property = get_dtype_property (dtype_);

  auto blocks = (w_ + dtype_property.items_per_block - 1)
                / dtype_property.items_per_block;

  ws_ = elem_bytes ();
  hs_ = blocks * dtype_property.bytes_per_block;
  cs_ = h_ * hs_;
}

ShaderConstants
Tensor::shape_constant () const
{
  ShapeConstant shape = { (uint32_t)c_,  (uint32_t)h_,  (uint32_t)w_,
                          (uint32_t)cs_, (uint32_t)hs_, (uint32_t)ws_ };

  return shape;
}

size_t
Tensor::size () const
{
  return c_ * h_ * w_;
}

absl::Status
Tensor::reshape (size_t const c, size_t const h, size_t const w)
{
  if (c * h * w != c_ * h_ * w_)
    {
      return absl::OutOfRangeError (absl::StrFormat (
          "reshape from (%zu, %zu, %zu) to (%zu, %zu, %zu) error.", c_, h_, w_,
          c, h, w));
    }

  c_ = c;
  h_ = h;
  w_ = w;

  update_strides_ ();
  return absl::OkStatus ();
}

Tensor
Tensor::view (size_t c, size_t h, size_t w)
{
  Tensor v = *this;
  v.c_ = c;
  v.h_ = h;
  v.w_ = w;

  return v;
}

VkBuffer &
Tensor::data ()
{
  return data_;
}

absl::Status
Tensor::flush ()
{
  if (!visable_)
    {
      return absl::UnimplementedError (
          absl::StrFormat ("cannot flush to invisable tensor"));
    }

  VkMappedMemoryRange range = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr,
                                mem_.deviceMemory, mem_.offset, mem_.size };

  auto ret = vkFlushMappedMemoryRanges (dev_->device (), 1, &range);
  if (ret != VK_SUCCESS)
    {
      return absl::InternalError (
          absl::StrFormat ("failed at flushing to tensor %d", int (ret)));
    }

  return absl::OkStatus ();
}

absl::Status
Tensor::invalid ()
{
  if (!visable_)
    {
      return absl::UnimplementedError (
          absl::StrFormat ("cannot invalid an invisable tensor"));
    }

  VkMappedMemoryRange range = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr,
                                mem_.deviceMemory, mem_.offset, mem_.size };

  auto ret = vkInvalidateMappedMemoryRanges (dev_->device (), 1, &range);
  if (ret != VK_SUCCESS)
    {
      return absl::InternalError (absl::StrFormat (
          "failed at invaliding tensor memory, ret = %d", int (ret)));
    }

  return absl::OkStatus ();
}

void
Tensor::release_ ()
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

Tensor::DType
Tensor::dtype () const
{
  return dtype_;
}

Tensor::~Tensor () { release_ (); }

Tensor
Tensor::like (const Tensor &tensor)
{
  Tensor tmp (tensor.channels (), tensor.height (), tensor.width (),
              tensor.dev_, tensor.dtype (), tensor.visable ());
  return tmp;
}
}

