#ifndef __VKLLAMA_TENSOR__
#define __VKLLAMA_TENSOR__

#include "allocator.h"
#include <atomic>
#include <vulkan/vulkan.h>

class GPUDevice;
class VkTensor
{
public:
    typedef enum
    {
        FP32
    } DType;

    static VkTensor like(VkTensor const&);
    VkTensor();
    VkTensor(const int c,
             const int h,
             const int w,
             GPUDevice* dev,
             const bool visable = false);

    VkTensor& operator=(VkTensor const&);
    VkTensor(const VkTensor& rhs);
    VkTensor(VkTensor&& rhs);

    ~VkTensor();

    size_t channels() const;
    size_t height() const;
    size_t width() const;
    size_t size() const;
    VkAccessFlags access_flags() const;
    VkPipelineStageFlags pipeline_stage() const;
    void set_access_flags(VkAccessFlags access_flags);
    void set_pipeline_stage(VkPipelineStageFlags stage_flags);

    VkBuffer& data();

    VkResult create();
    size_t bytes() const;
    bool visable() const;
    VkResult flush();
    VkResult invalid();

    void* host();

private:
    int c_;
    int h_;
    int w_;
    size_t bytes_;

    GPUDevice* dev_;
    bool visable_;
    VkBuffer data_;
    Allocator::MemBlock mem_;
    struct __TensorStatus
    {
        std::atomic<VkAccessFlags> access_flags_;
        std::atomic<VkPipelineStageFlags> pipeline_stage_;
        std::atomic<int> ref_;
    };

    __TensorStatus* status_;
};

#endif
