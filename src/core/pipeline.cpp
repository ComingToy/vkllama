#include "pipeline.h"
#include "absl/strings/str_format.h"
#include "tensor.h"
#include <array>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace vkllama
{
Pipeline::Pipeline (GPUDevice *device, const uint8_t *spv,
                    const size_t spv_size,
                    ShaderConstants const &specialization,
                    ShaderInfo const &info)
    : init_ (false), module_ (VK_NULL_HANDLE), spv_ (spv),
      spv_size_ (spv_size), shaderInfo_ (info),
      specialization_ (specialization), device_ (device),
      descriptorSetLayout_ (VK_NULL_HANDLE), descriptorPool_ (VK_NULL_HANDLE),
      descriptorSet_ (VK_NULL_HANDLE), pipelineLayout_ (VK_NULL_HANDLE),
      pipeline_ (VK_NULL_HANDLE), x_ (0), y_ (0), z_ (0),
      queryPool_ (VK_NULL_HANDLE), descriptor_update_template_ (VK_NULL_HANDLE)
{
}

Pipeline::~Pipeline ()
{
  vkDestroyPipeline (device_->device (), pipeline_, nullptr);
  vkDestroyShaderModule (device_->device (), module_, nullptr);
  vkDestroyPipelineLayout (device_->device (), pipelineLayout_, nullptr);
  vkDestroyDescriptorSetLayout (device_->device (), descriptorSetLayout_,
                                nullptr);
  vkDestroyDescriptorPool (device_->device (), descriptorPool_, nullptr);
  vkDestroyQueryPool (device_->device (), queryPool_, nullptr);
  vkDestroyDescriptorUpdateTemplate (device_->device (),
                                     descriptor_update_template_, nullptr);
}

absl::Status
Pipeline::init ()
{
  auto ret = limits_ ();
  if (!ret.ok ())
    return ret;

  ret = create_shader_module_ ();
  if (!ret.ok ())
    return ret;

  ret = create_pipeline_layout_ ();
  if (!ret.ok ())
    return ret;

  ret = create_descriptor_set_ ();
  if (!ret.ok ())
    {
      return ret;
    }

  ret = create_query_pool_ ();
  if (!ret.ok () && !absl::IsUnimplemented (ret))
    {
      return ret;
    }

  ret = create_pipeline_ (specialization_);
  if (!ret.ok ())
    {
      return ret;
    }

  ret = create_descriptor_update_template_ ();
  if (!ret.ok ())
    {
      return ret;
    }

  init_ = true;
  return absl::OkStatus ();
}

absl::Status
Pipeline::create_shader_module_ ()
{

  VkShaderModuleCreateInfo createInfo
      = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, spv_size_,
          reinterpret_cast<const uint32_t *> (spv_) };

  auto ret = vkCreateShaderModule (device_->device (), &createInfo, nullptr,
                                   &module_);
  if (ret != VK_SUCCESS)
    {
      return absl::InternalError (
          absl::StrFormat ("create shader module failed: %d", int (ret)));
    }

  return absl::OkStatus ();
}

absl::Status
Pipeline::create_pipeline_layout_ ()
{
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  for (int i = 0; i < shaderInfo_.binding_count; ++i)
    {
      uint32_t binding = i;
      // auto type = shaderInfo_.binding_types[i];
      VkDescriptorSetLayoutBinding b
          = { binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
              VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
      bindings.push_back (b);
    }

  {
    VkDescriptorSetLayoutCreateInfo createInfo
        = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0,
            (uint32_t)bindings.size (), bindings.data () };

    auto ret = vkCreateDescriptorSetLayout (device_->device (), &createInfo,
                                            nullptr, &descriptorSetLayout_);
    if (ret != VK_SUCCESS)
      return absl::InternalError (absl::StrFormat (
          "create descriptors set layout failed: %d", int (ret)));
  }

  {
    uint32_t size = shaderInfo_.push_constant_bytes;
    VkPushConstantRange range = { VK_SHADER_STAGE_COMPUTE_BIT, 0, size };

    VkPipelineLayoutCreateInfo createInfo
        = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            1,
            &descriptorSetLayout_,
            shaderInfo_.push_constant_bytes > 0 ? 1u : 0u,
            &range };
    auto ret = vkCreatePipelineLayout (device_->device (), &createInfo,
                                       nullptr, &pipelineLayout_);
    if (ret != VK_SUCCESS)
      {
        return absl::InternalError (
            absl::StrFormat ("create pipeline layout failed: %d", int (ret)));
      }
  }

  return absl::OkStatus ();
}

absl::Status
Pipeline::create_pipeline_ (ShaderConstants const &constants)
{
  const int n = constants.elem_num ();
  std::vector<VkSpecializationMapEntry> mapEntries;
  for (int i = 0; i < n - 3; ++i)
    {
      uint32_t offset = constants.offsets ()[i];
      uint32_t size = constants.sizes ()[i];

      VkSpecializationMapEntry entry = { (uint32_t)i, offset, size };
      mapEntries.push_back (entry);
    }

  for (int i = n - 3, k = 0; i < n; ++i, ++k)
    {
      uint32_t id = 253 + k;
      uint32_t offset = constants.offsets ()[i];
      uint32_t size = constants.sizes ()[i];
      mapEntries.push_back (VkSpecializationMapEntry{ id, offset, size });
    }

  VkSpecializationInfo specializationInfo
      = { (uint32_t)mapEntries.size (), mapEntries.data (), constants.bytes (),
          constants.data () };

  VkPipelineShaderStageCreateInfo shaderCreateInfo
      = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          nullptr,
          0,
          VK_SHADER_STAGE_COMPUTE_BIT,
          module_,
          "main",
          &specializationInfo };

  VkComputePipelineCreateInfo computePipelineCreateInfo
      = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
          nullptr,
          0,
          shaderCreateInfo,
          pipelineLayout_,
          VK_NULL_HANDLE,
          0 };

  auto ret = vkCreateComputePipelines (device_->device (), 0, 1,
                                       &computePipelineCreateInfo, nullptr,
                                       &pipeline_);
  if (ret != VK_SUCCESS)
    {
      return absl::InternalError (
          absl::StrFormat ("create compute pipeline failed: %d", int (ret)));
    }

  return absl::OkStatus ();
}

absl::Status
Pipeline::create_descriptor_set_ ()
{
  VkDescriptorPoolSize size;
  size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  size.descriptorCount = 1024; // large enough?

  VkDescriptorPoolCreateInfo createInfo
      = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
          nullptr,
          VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
          1,
          1,
          &size };

  auto ret = vkCreateDescriptorPool (device_->device (), &createInfo, nullptr,
                                     &descriptorPool_);
  if (ret != VK_SUCCESS)
    {
      return absl::InternalError (
          absl::StrFormat ("failed at create descriptors pool %d", int (ret)));
    }

  VkDescriptorSetAllocateInfo allocInfo
      = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
          descriptorPool_, 1, &descriptorSetLayout_ };

  ret = vkAllocateDescriptorSets (device_->device (), &allocInfo,
                                  &descriptorSet_);
  if (ret != VK_SUCCESS)
    {
      return absl::InternalError (
          absl::StrFormat ("allocate descriptors set failed: %d", int (ret)));
    }

  return absl::OkStatus ();
}

absl::Status
Pipeline::create_descriptor_update_template_ ()
{
  if (!device_->support_descriptor_templ_update ())
    {
      return absl::OkStatus ();
    }

  std::vector<VkDescriptorUpdateTemplateEntry> entries (
      shaderInfo_.binding_count);

  uint32_t offset = 0;
  for (uint32_t i = 0; i < shaderInfo_.binding_count; ++i)
    {
      entries[i] = { i,      0,
                     1,      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                     offset, sizeof (VkDescriptorBufferInfo) };
      offset += sizeof (VkDescriptorBufferInfo);
    }

  VkDescriptorUpdateTemplateCreateInfo info
      = { VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO,
          nullptr,
          0,
          (uint32_t)shaderInfo_.binding_count,
          entries.data (),
          VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET,
          descriptorSetLayout_,
          VK_PIPELINE_BIND_POINT_COMPUTE,
          pipelineLayout_,
          0 };

  auto ret = vkCreateDescriptorUpdateTemplate (
      device_->device (), &info, nullptr, &descriptor_update_template_);
  if (ret != VK_SUCCESS)
    {
      return absl::InternalError (absl::StrFormat (
          "create descriptor update template failed: %d", int (ret)));
    }

  return absl::OkStatus ();
}

absl::Status
Pipeline::create_query_pool_ ()
{
  if (!device_->support_pipeline_statistics ())
    {
      return absl::UnimplementedError ("pipeline statistics is unsupported.");
    }

  VkQueryPoolCreateInfo createInfo
      = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
          nullptr,
          0,
          VK_QUERY_TYPE_TIMESTAMP,
          2,
          0 };
  auto ret = vkCreateQueryPool (device_->device (), &createInfo, nullptr,
                                &queryPool_);
  if (ret != VK_SUCCESS)
    {
      return absl::InternalError (
          absl::StrFormat ("create query pool failed: %d", int (ret)));
    }

  return absl::OkStatus ();
}

absl::Status
Pipeline::update_bindings (std::vector<Tensor> bindings)
{
  return set_bindings_ (bindings);
}

absl::Status
Pipeline::update_bindings (std::vector<Tensor> bindings,
                           const std::vector<uint32_t> &indices)
{
  return set_bindings_ (bindings, indices);
}

absl::Status
Pipeline::set_bindings_ (std::vector<Tensor> bindings)
{
  std::vector<VkDescriptorBufferInfo> descriptors;
  descriptors.resize (bindings.size ());
  for (size_t i = 0; i < bindings.size (); ++i)
    {
      descriptors[i] = { bindings[i].data (), 0, VK_WHOLE_SIZE };
    }

  if (device_->support_descriptor_templ_update ())
    {
      vkUpdateDescriptorSetWithTemplate (device_->device (), descriptorSet_,
                                         descriptor_update_template_,
                                         descriptors.data ());
    }
  else
    {
      std::vector<VkWriteDescriptorSet> writes;
      writes.resize (bindings.size ());
      for (size_t i = 0; i < bindings.size (); ++i)
        {
          writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        nullptr,
                        descriptorSet_,
                        (uint32_t)i,
                        0,
                        1,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        nullptr,
                        &descriptors[i],
                        nullptr };
        }
      vkUpdateDescriptorSets (device_->device (), writes.size (),
                              writes.data (), 0, nullptr);
    }

  return absl::OkStatus ();
}

absl::Status
Pipeline::set_bindings_ (std::vector<Tensor> bindings,
                         const std::vector<uint32_t> &indices)
{
  std::vector<VkDescriptorBufferInfo> descriptors (indices.size ());
  descriptors.resize (indices.size ());

  std::vector<VkWriteDescriptorSet> writes;
  writes.resize (bindings.size ());
  for (size_t i = 0; i < indices.size (); ++i)
    {
      descriptors[i] = { bindings[i].data (), 0, VK_WHOLE_SIZE };
      writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    nullptr,
                    descriptorSet_,
                    indices[i],
                    0,
                    1,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    nullptr,
                    &descriptors[i],
                    nullptr };
    }

  vkUpdateDescriptorSets (device_->device (), writes.size (), writes.data (),
                          0, nullptr);

  return absl::OkStatus ();
}

VkPipeline &
Pipeline::vkpileine ()
{
  return pipeline_;
}

VkDescriptorSet &
Pipeline::vkdescriptorset ()
{
  return descriptorSet_;
}

VkPipelineLayout &
Pipeline::vklayout ()
{
  return pipelineLayout_;
}

VkQueryPool &
Pipeline::vkquerypool ()
{
  return queryPool_;
}

void
Pipeline::query_exec_timestamp ()
{
  if (!device_->support_pipeline_statistics ())
    {
      return;
    }
  vkGetQueryPoolResults (device_->device (), queryPool_, 0, 2,
                         time_stamps_.size () * sizeof (uint64_t),
                         time_stamps_.data (), sizeof (uint64_t),
                         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
}

uint64_t
Pipeline::time ()
{

  auto delta = (time_stamps_[1] - time_stamps_[0])
               * device_->timestamp_period () / 1000.0f;
  return (uint64_t)delta;
}

absl::Status
Pipeline::limits_ ()
{
  auto const &limits = device_->limits ();
  shaderInfo_.local_x
      = std::min (shaderInfo_.local_x, limits.maxComputeWorkGroupSize[0]);
  shaderInfo_.local_y
      = std::min (shaderInfo_.local_y, limits.maxComputeWorkGroupSize[1]);
  shaderInfo_.local_z
      = std::min (shaderInfo_.local_z, limits.maxComputeWorkGroupSize[2]);
  auto group_size
      = shaderInfo_.local_x * shaderInfo_.local_y * shaderInfo_.local_z;
  if (group_size > limits.maxComputeWorkGroupInvocations)
    {
      return absl::OutOfRangeError (absl::StrFormat (
          "size of thread group is too large. max compute work group "
          "invocations = %zu, but %zu provided",
          (size_t)limits.maxComputeWorkGroupInvocations, (size_t)group_size));
    }

  specialization_.push_back (shaderInfo_.local_x);
  specialization_.push_back (shaderInfo_.local_y);
  specialization_.push_back (shaderInfo_.local_z);

  return absl::OkStatus ();
}

uint32_t
Pipeline::group_x () const
{
  return x_;
}

uint32_t
Pipeline::group_y () const
{
  return y_;
}

uint32_t
Pipeline::group_z () const
{
  return z_;
}

absl::Status
Pipeline::set_group (uint32_t x, uint32_t y, uint32_t z)
{
  x_ = x;
  y_ = y;
  z_ = z;

  auto const &limits = device_->limits ();

  if (group_x () > limits.maxComputeWorkGroupCount[0]
      || group_y () > limits.maxComputeWorkGroupCount[1]
      || group_z () > limits.maxComputeWorkGroupCount[2])
    {
      return absl::OutOfRangeError (absl::StrFormat (
          "group size out of range. (%zu, %zu, %zu) given but limit is (%zu, "
          "%zu, %zu)",
          (size_t)group_x (), (size_t)group_y (), (size_t)group_z (),
          (size_t)limits.maxComputeWorkGroupCount[0],
          (size_t)limits.maxComputeWorkGroupCount[1],
          (size_t)limits.maxComputeWorkGroupCount[2]));
    }
  return absl::OkStatus ();
}

Pipeline::ShaderInfo const &
Pipeline::shader_info () const
{
  return shaderInfo_;
}
}

