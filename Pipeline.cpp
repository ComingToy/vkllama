#include "Pipeline.h"
#include <vector>
#include <vulkan/vulkan_core.h>

Pipeline::Pipeline(GPUDevice* device,
                   std::string const& f,
                   std::vector<SpecializationType> const& specialization,
                   const int x,
                   const int y,
                   const int z,
                   ShaderInfo const& info)
  : shaderInfo_(info)
  , device_(device)
  , x_(x)
  , y_(y)
  , z_(z)
{
    create_pipeline_layout_();
}

VkResult
Pipeline::create_pipeline_layout_()
{
    for (int i = 0; i < shaderInfo_.binding_count; ++i) {
        uint32_t binding = i;
        auto type = shaderInfo_.binding_types[i];
        VkDescriptorSetLayoutBinding b = { binding,
                                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                           1,
                                           VK_SHADER_STAGE_COMPUTE_BIT,
                                           nullptr };
        bindings_.push_back(b);
    }

    {
        VkDescriptorSetLayoutCreateInfo createInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            (uint32_t)bindings_.size(),
            bindings_.data()
        };

        auto ret = vkCreateDescriptorSetLayout(
          device_->device(), &createInfo, nullptr, &descriptorSetLayout_);
        if (ret != VK_SUCCESS)
            return ret;
    }

    {
        std::vector<VkPushConstantRange> constants;
        for (int i = 0; i < shaderInfo_.push_constant_count; ++i) {
            uint32_t offset = i * sizeof(PushConstantType);
            uint32_t size = sizeof(PushConstantType);
            constants.emplace_back(VK_SHADER_STAGE_COMPUTE_BIT, offset, size);
        }

        VkPipelineLayoutCreateInfo createInfo = {
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            1,
            &descriptorSetLayout_,
            (uint32_t)constants.size(),
            constants.data()
        };
        return vkCreatePipelineLayout(
          device_->device(), &createInfo, nullptr, &pipelineLayout_);
    }
}

VkResult
Pipeline::create_pipeline_(std::vector<SpecializationType> const& constants)
{
    const int n = constants.size();
    std::vector<VkSpecializationMapEntry> mapEntries;
    for (int i = 0; i < n; ++i) {
        uint32_t offset = i * sizeof(SpecializationType);
        uint32_t size = sizeof(SpecializationType);
        mapEntries.emplace_back((uint32_t)i, offset, size);
    }

    VkSpecializationInfo specializationInfo = { (uint32_t)mapEntries.size(),
                                                mapEntries.data(),
                                                constants.size() *
                                                  sizeof(SpecializationType),
                                                constants.data() };

    VkPipelineShaderStageCreateInfo shaderCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr,
        0,
        VK_SHADER_STAGE_COMPUTE_BIT,
        module_,
        "vkllama",
        &specializationInfo
    };

    VkComputePipelineCreateInfo computePipelineCreateInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        nullptr,
        0,
        shaderCreateInfo,
        pipelineLayout_,
        VK_NULL_HANDLE,
        0
    };

    auto ret = vkCreateComputePipelines(
      device_->device(), 0, 1, &computePipelineCreateInfo, nullptr, &pipeline_);
}
