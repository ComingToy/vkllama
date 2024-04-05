#ifndef __VKLLAMA_REDUCE_H__
#define __VKLLAMA_REDUCE_H__

#include "shaders/vkllama_comp_shaders.h"
#include "src/core/command.h"
#include "src/core/gpu_device.h"
#include "src/core/pipeline.h"
#include "src/core/tensor.h"
#include "src/ops/op.h"
#include <memory>

// 0: sum 1: max 2: min 3: mean
class Reduce : public Op
{
public:
  Reduce (GPUDevice *gpu, Command *command, int op_type)
      : Op (gpu, command), op_type_ (op_type)
  {
  }

  VkResult
  init () override
  {
    Pipeline::ShaderInfo stage0Info = { 1, 1, 3, 64, 4, 1 };
    Pipeline::ShaderInfo stage1Info = { 1, 2, 5, 1, 64, 1 };

    Pipeline::ConstantType op_type = { .i = op_type_ == 3 ? 0 : op_type_ };
    stage0_.reset (new Pipeline (dev_, __get_reduce_stage0_comp_spv_code (),
                                 __get_reduce_stage0_comp_spv_size (),
                                 { op_type }, stage0Info));

    stage1_.reset (new Pipeline (dev_, __get_reduce_stage1_comp_spv_code (),
                                 __get_reduce_stage1_comp_spv_size (),
                                 { op_type }, stage1Info));

    VkResult ret = stage0_->init ();
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    return stage1_->init ();
  }

  uint64_t
  time () override
  {
    return stage0_->time () + stage1_->time ();
  };

  VkResult
  operator() (VkTensor a, VkTensor &b)
  {
    b = VkTensor (a.channels (), a.height (), 1, dev_);
    VkResult ret = b.create ();
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    uint32_t group_x = (a.width () + 63) / 64, group_y = (a.height () + 3) / 4,
             group_z = a.channels ();
    ret = stage0_->set_group (group_x, group_y, group_z);
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    Pipeline::ConstantType C = { .i = static_cast<int> (a.channels ()) };
    Pipeline::ConstantType H = { .i = static_cast<int> (a.height ()) };
    Pipeline::ConstantType W = { .i = static_cast<int> (a.width ()) };
    ret = command_->record_pipeline (*stage0_, { a }, { C, H, W });
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    a.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
    a.set_pipeline_stage (VK_SHADER_STAGE_COMPUTE_BIT);

    Pipeline::ConstantType mean_scale;
    if (op_type_ == 3)
      {
        mean_scale.f = 1.0f / static_cast<float> (a.width ());
      }
    else
      {
        mean_scale.f = 1.0f;
      }

    Pipeline::ConstantType G = { .i = (int)group_x };
    ret = stage1_->set_group (1, group_y, group_z);
    ret = command_->record_pipeline (*stage1_, { a, b },
                                     { C, H, W, G, mean_scale });
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    b.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
    b.set_pipeline_stage (VK_SHADER_STAGE_COMPUTE_BIT);
    return VK_SUCCESS;
  }

private:
  std::unique_ptr<Pipeline> stage0_;
  std::unique_ptr<Pipeline> stage1_;
  const int op_type_;
};

#endif
