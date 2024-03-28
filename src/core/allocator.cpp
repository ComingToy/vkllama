#include "allocator.h"
#include <set>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

Allocator::Allocator(GPUDevice* dev)
    : dev_(dev)
{
}

Allocator::~Allocator()
{
    std::map<VkDeviceMemory, VkMemoryPropertyFlags> mems;
    for (auto kv : mem_pool_)
    {
        for (auto const& b : kv.second)
        {
            mems[b.mem] = b.flags;
        }
    }

    std::for_each(mems.begin(), mems.end(), [this](auto& item) {
        if (item.second & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        {
            vkUnmapMemory(dev_->device(), item.first);
        }
        vkFreeMemory(dev_->device(), item.first, nullptr);
    });
}

VkResult
Allocator::allocate(VkMemoryRequirements const& req,
                    MemBlock* block,
                    bool visable)
{
    __MemBlock _block;

    auto ret = allocate_(req,
                         visable ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                 : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         _block);

    if (ret != VK_SUCCESS)
        return ret;

    block->mem = _block.mem;
    block->offset = _block.offset;
    block->type = _block.type;
    block->size = _block.size;
    block->host = _block.host;
    block->align = _block.align;
    return VK_SUCCESS;
}

void Allocator::free(const MemBlock block)
{
    for (auto pos = mem_pool_[block.type].begin();
         pos != mem_pool_[block.type].end();
         ++pos)
    {
        if (pos->mem == block.mem && pos->offset == block.offset)
        {
            if (!pos->parent || pos->parent->allocated)
            {
                pos->allocated = false;
                return;
            }

            if (pos->parent)
            {
                pos->parent->offset = pos->offset;
                pos->parent->size += pos->size;
                pos->parent->host = pos->host;
                mem_pool_[block.type].erase(pos);
                return;
            }
        }
    }
}

VkResult
Allocator::allocate_from_pool_(const uint32_t type,
                               const VkDeviceSize req,
                               const VkDeviceSize align,
                               __MemBlock& result)
{
    if (mem_pool_.find(type) == mem_pool_.cend())
    {
        return VK_ERROR_OUT_OF_POOL_MEMORY;
    }

    auto size = (req + align - 1) / align * align;
    auto& blocks = mem_pool_[type];
    for (auto pos = blocks.begin(); pos != blocks.end(); ++pos)
    {
        auto& block = *pos;
        if (block.allocated || block.size < size)
            continue;

        result = {block.mem, size, block.offset, align, block.flags,
                  block.type, &block, block.host, true};

        block.size -= size;
        block.offset += size;
        block.host = (uint8_t*)block.host + size;

        blocks.insert(pos, result);
        return VK_SUCCESS;
    }

    return VK_ERROR_OUT_OF_POOL_MEMORY;
}

VkResult
Allocator::allocate_(VkMemoryRequirements const& req,
                     VkMemoryPropertyFlags const flags,
                     __MemBlock& result)
{
    uint32_t type = dev_->find_mem(req.memoryTypeBits, flags);

    auto ret = allocate_from_pool_(type, req.size, req.alignment, result);
    if (ret == VK_SUCCESS || ret != VK_ERROR_OUT_OF_POOL_MEMORY)
    {
        return ret;
    }

    {
        auto size = ((req.size + BLOCK_ALIGN - 1) / BLOCK_ALIGN) * BLOCK_ALIGN;
        VkMemoryAllocateInfo allocInfo = {
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, size, type};

        VkDeviceMemory mem;
        auto ret = vkAllocateMemory(dev_->device(), &allocInfo, nullptr, &mem);
        if (ret != VK_SUCCESS)
            return ret;

        void* host = nullptr;
        if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        {
            ret = vkMapMemory(dev_->device(), mem, 0, VK_WHOLE_SIZE, 0, &host);
            if (ret != VK_SUCCESS)
            {
                vkFreeMemory(dev_->device(), mem, nullptr);
                return ret;
            }
        }

        auto new_block = __MemBlock{mem, size, 0, 0, flags, type, nullptr, host, false};
        mem_pool_[type].push_back(new_block);
    }

    return allocate_from_pool_(type, req.size, req.alignment, result);
}
