#ifndef __LLAMA_ALLOCATOR_H__
#define __LLAMA_ALLOCATOR_H__
#include "gpu_device.h"
#include <list>
#include <map>

class Allocator
{
public:
    Allocator() = delete;
    friend GPUDevice;

    struct MemBlock
    {
        VkDeviceMemory mem;
        VkDeviceSize offset;
        VkDeviceSize size;
        VkDeviceSize align;
        void* host;
        uint32_t type;
    };

    VkResult allocate(VkMemoryRequirements const& req,
                      MemBlock* block,
                      bool visable = false);
    void free(const MemBlock block);

private:
    Allocator(GPUDevice* dev);
    ~Allocator();
    GPUDevice* dev_;

    struct __MemBlock
    {
        VkDeviceMemory mem;
        VkDeviceSize size;
        VkDeviceSize offset;
        VkDeviceSize align;
        VkMemoryPropertyFlags flags;
        uint32_t type;
        __MemBlock* parent;
        void* host;
        bool allocated;
    };

    VkResult allocate_(VkMemoryRequirements const& req,
                       const VkMemoryPropertyFlags flags,
                       __MemBlock& block);
    VkResult allocate_from_pool_(const uint32_t type,
                                 const VkDeviceSize size,
                                 const VkDeviceSize align,
                                 __MemBlock& result);

    std::map<VkBuffer, VkDeviceMemory> allocated_;
    // <type, memblock>
    std::map<uint32_t, std::list<__MemBlock> > mem_pool_;
    static constexpr VkDeviceSize BLOCK_ALIGN = 1024 * 1024 * 10;
};

#endif
