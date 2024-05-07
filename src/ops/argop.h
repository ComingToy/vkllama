#ifndef __VKLLAMA_ARGOP_H__
#define __VKLLAMA_ARGOP_H__

#include "src/core/command.h"
#include "src/ops/op.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <memory>
#include <vector>

template <int op_type> class ArgOp : public Op
{
public:
  ArgOp (GPUDevice *gpu, Command *command) : Op (gpu, command) {}
  VkResult
  init () noexcept override
  {
    Pipeline::ShaderInfo info0 = { 1, 3, 3, 32, 4, 1 };
    Pipeline::ShaderInfo info1 = { 1, 3, 3, 1, 128, 1 };

    pipeline0_.reset (new Pipeline (dev_, __get_argmax_stage0_comp_spv_code (),
                                    __get_argmax_stage0_comp_spv_size (),
                                    { { .i = op_type } }, info0));
    pipeline1_.reset (new Pipeline (dev_, __get_argmax_stage1_comp_spv_code (),
                                    __get_argmax_stage1_comp_spv_size (),
                                    { { .i = op_type } }, info1));

    auto ret = pipeline0_->init ();
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    return pipeline1_->init ();
  }

  VkResult
  operator() (VkTensor in, VkTensor &out) noexcept
  {
    out = VkTensor (in.channels (), in.height (), 1, dev_, VkTensor::UINT32);
    auto ret = out.create ();
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    uint32_t group_x = (in.width () + 31) / 32,
             group_y = (in.height () + 3) / 4, group_z = in.channels ();

    if ((ret = pipeline0_->set_group (group_x, group_y, group_z))
        != VK_SUCCESS)
      {
        return ret;
      }

    stage0_values_ = VkTensor (in.channels (), in.height (), group_x, dev_);
    stage0_indices_ = VkTensor (in.channels (), in.height (), group_x, dev_,
                                VkTensor::UINT32);

    if ((ret = stage0_values_.create ()) != VK_SUCCESS
        || (ret = stage0_indices_.create ()) != VK_SUCCESS)
      {
        return ret;
      }

    std::vector<Pipeline::ConstantType> shape
        = { { .u32 = static_cast<uint32_t> (in.channels ()) },
            { .u32 = static_cast<uint32_t> (in.height ()) },
            { .u32 = static_cast<uint32_t> (in.width ()) } };

    ret = command_->record_pipeline (
        *pipeline0_, { in, stage0_indices_, stage0_values_ }, shape);
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    stage0_indices_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
    stage0_indices_.set_pipeline_stage (VK_SHADER_STAGE_COMPUTE_BIT);
    stage0_values_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
    stage0_values_.set_pipeline_stage (VK_SHADER_STAGE_COMPUTE_BIT);

    ret = pipeline1_->set_group (1, group_y, group_z);
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    shape[2].u32 = group_x;

    ret = command_->record_pipeline (
        *pipeline1_, { stage0_indices_, stage0_values_, out }, shape);

    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    out.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
    out.set_pipeline_stage (VK_SHADER_STAGE_COMPUTE_BIT);

    return VK_SUCCESS;
  }

  uint64_t
  time () noexcept override
  {
    return pipeline0_->time () + pipeline1_->time ();
  }

private:
  std::unique_ptr<Pipeline> pipeline0_;
  std::unique_ptr<Pipeline> pipeline1_;
  VkTensor stage0_values_;
  VkTensor stage0_indices_;
};

using ArgMax = ArgOp<0>;
using ArgMin = ArgOp<1>;

#endif
