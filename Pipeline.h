#ifndef __VKLLAMA_CPP_PIPELINE_H__
#define __VKLLAMA_CPP_PIPELINE_H__
#include "GPUDevice.h"
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

class Pipeline
{
  public:
    struct ShaderInfo
    {
        int specialization_count;
        int binding_count;
        int push_constant_count;

        // 0 = null
        // 1 = storage buffer
        // 2 = storage image
        // 3 = combined image sampler
        int binding_types[8];
    };

    union PushConstantType
    {
        int i;
        float f;
    } __attribute__((packed));

    union SpecializationType
    {
        int i;
        float f;
        uint32_t u32;
    } __attribute__((packed));

    Pipeline(GPUDevice* device_,
             std::string const& shader,
             std::vector<SpecializationType> const& specialization,
             const int x,
             const int y,
             const int z,
             ShaderInfo const& info);

  private:
    VkShaderModule module_;
    VkDescriptorSetLayout setLayout_;
    std::vector<VkDescriptorSetLayoutBinding> bindings_;
    ShaderInfo shaderInfo_;
    GPUDevice* device_;
    VkDescriptorSetLayout descriptorSetLayout_;
    VkPipelineLayout pipelineLayout_;
    VkPipeline pipeline_;
    const int x_;
    const int y_;
    const int z_;

    VkResult create_pipeline_layout_();
    VkResult create_pipeline_(std::vector<SpecializationType> const&);
};
#endif
