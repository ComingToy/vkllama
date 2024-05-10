#ifndef __VKLLAMA_CPP_PIPELINE_H__
#define __VKLLAMA_CPP_PIPELINE_H__
#include "gpu_device.h"
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

class VkTensor;
class Pipeline
{
public:
  struct ShaderInfo
  {
    int specialization_count;
    int binding_count;
    int push_constant_count;
    uint32_t local_x, local_y, local_z;

    // 0 = null
    // 1 = storage buffer
    // 2 = storage image
    // 3 = combined image sampler
    int binding_types[8];
  };

  union ConstantType
  {
    int32_t i;
    float f;
    uint32_t u32;
  } __attribute__ ((packed));

  Pipeline (GPUDevice *device_, const uint8_t *spv, const size_t spv_size,
            std::vector<ConstantType> const &specialization,
            ShaderInfo const &info);

  ~Pipeline ();

  VkResult init ();
  VkPipeline &vkpileine ();
  VkDescriptorSet &vkdescriptorset ();
  VkPipelineLayout &vklayout ();
  VkQueryPool &vkquerypool ();
  uint64_t time ();

  uint32_t group_x () const;
  uint32_t group_y () const;
  uint32_t group_z () const;
  VkResult set_group (uint32_t x, uint32_t y, uint32_t z);
  VkResult update_bindings (std::vector<VkTensor> bindings);

private:
  bool init_;
  VkShaderModule module_;
  const uint8_t *spv_;
  const size_t spv_size_;
  ShaderInfo shaderInfo_;
  std::vector<ConstantType> specialization_;

  GPUDevice *device_;
  VkDescriptorSetLayout descriptorSetLayout_;
  VkDescriptorPool descriptorPool_;
  VkDescriptorSet descriptorSet_;
  VkPipelineLayout pipelineLayout_;
  VkPipeline pipeline_;
  int x_;
  int y_;
  int z_;
  VkQueryPool queryPool_;
  VkDescriptorUpdateTemplate descriptor_update_template_;

  VkResult create_shader_module_ ();
  VkResult create_pipeline_layout_ ();
  VkResult create_descriptor_set_ ();
  VkResult create_pipeline_ (std::vector<ConstantType> const &);
  VkResult create_query_pool_ ();
  VkResult create_descriptor_update_template_ ();
  VkResult set_bindings_ (std::vector<VkTensor> bindings);
  VkResult limits_ ();
};
#endif
