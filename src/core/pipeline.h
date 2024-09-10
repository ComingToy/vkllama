#ifndef __VKLLAMA_CPP_PIPELINE_H__
#define __VKLLAMA_CPP_PIPELINE_H__
#include "gpu_device.h"
#include "shader_constants.h"
#include <array>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

namespace vkllama
{
class Tensor;
class Pipeline
{
public:
  struct ShaderInfo
  {
    int specialization_count;
    int binding_count;
    uint32_t push_constant_bytes;
    uint32_t local_x, local_y, local_z;
  };

  Pipeline (GPUDevice *device_, const uint8_t *spv, const size_t spv_size,
            ShaderConstants const &specialization, ShaderInfo const &info);

  ~Pipeline ();

  absl::Status init ();
  VkPipeline &vkpileine ();
  VkDescriptorSet &vkdescriptorset ();
  VkPipelineLayout &vklayout ();
  VkQueryPool &vkquerypool ();
  uint64_t time ();

  uint32_t group_x () const;
  uint32_t group_y () const;
  uint32_t group_z () const;
  absl::Status set_group (uint32_t x, uint32_t y, uint32_t z);
  absl::Status update_bindings (std::vector<Tensor> bindings);
  absl::Status update_bindings (std::vector<Tensor> bindings,
                                std::vector<uint32_t> const &indices);

  ShaderInfo const &shader_info () const;
  void query_exec_timestamp ();

private:
  bool init_;
  VkShaderModule module_;
  const uint8_t *spv_;
  const size_t spv_size_;
  ShaderInfo shaderInfo_;
  ShaderConstants specialization_;

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
  std::array<uint64_t, 2> time_stamps_;

  absl::Status create_shader_module_ ();
  absl::Status create_pipeline_layout_ ();
  absl::Status create_descriptor_set_ ();
  absl::Status create_pipeline_ (ShaderConstants const &);
  absl::Status create_query_pool_ ();
  absl::Status create_descriptor_update_template_ ();
  absl::Status set_bindings_ (std::vector<Tensor> bindings);
  absl::Status set_bindings_ (std::vector<Tensor> bindings,
                              std::vector<uint32_t> const &indices);
  absl::Status limits_ ();
};
}

#endif
