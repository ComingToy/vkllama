#include "src/ops/softmax.h"
#include "src/core/command.h"
#include "src/core/gpu_device.h"
#include "src/core/pipeline.h"
#include "src/core/tensor.h"
#include "src/ops/reduce.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <vector>

Softmax::Softmax (GPUDevice *dev, Command *command) : Op (dev, command) {}
VkResult
Softmax::init ()
{
  reduce_.reset (new Reduce (dev_, command_, 1));
  auto ret = reduce_->init ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  Pipeline::ShaderInfo info0 = { 0, 4, 3, 32, 4, 1 };
  Pipeline::ShaderInfo info1 = { 0, 3, 4, 32, 4, 1 };

  softmax0_.reset (new Pipeline (dev_, __get_softmax_stage0_comp_spv_code (),
                                 __get_softmax_stage0_comp_spv_size (), {},
                                 info0));

  softmax1_.reset (new Pipeline (dev_, __get_softmax_stage1_comp_spv_code (),
                                 __get_softmax_stage1_comp_spv_size (), {},
                                 info1));

  if ((ret = softmax0_->init ()) != VK_SUCCESS
      || (ret = softmax1_->init ()) != VK_SUCCESS)
    {
      return ret;
    }

  return VK_SUCCESS;
}

uint64_t
Softmax::time ()
{
  return reduce_->time () + softmax0_->time () + softmax1_->time ();
}

VkResult
Softmax::operator() (VkTensor a, VkTensor &b)
{
  auto ret = reduce_->operator() (a, bias_);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  std::vector<Pipeline::ConstantType> shape
      = { { .u32 = (uint32_t)a.channels () },
          { .u32 = (uint32_t)a.height () },
          { .u32 = (uint32_t)a.width () } };

  uint32_t group_x = (a.width () + 31) / 32, group_y = (a.height () + 3) / 4,
           group_z = a.channels ();
  ret = softmax0_->set_group (group_x, group_y, group_z);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  m_ = VkTensor (a.channels (), a.height (), group_x, dev_);
  if ((ret = m_.create ()) != VK_SUCCESS)
    {
      return ret;
    }

  exps_ = VkTensor::like (a);
  if ((ret = exps_.create ()) != VK_SUCCESS)
    {
      return ret;
    }

  ret = command_->record_pipeline (*softmax0_, { a, bias_, m_, exps_ }, shape);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  m_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  m_.set_pipeline_stage (VK_SHADER_STAGE_COMPUTE_BIT);
  exps_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  exps_.set_pipeline_stage (VK_SHADER_STAGE_COMPUTE_BIT);
#if 0
b = VkTensor::like (a);
  auto ret = b.create ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  shape.push_back ({ .u32 = (uint32_t)m_.width () });
  softmax1_->set_group (group_x, group_y, group_z);
  ret = command_->record_pipeline (*softmax1_, { m_, exps_, b }, shape);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  b.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  b.set_pipeline_stage (VK_SHADER_STAGE_COMPUTE_BIT);
#endif
  b = m_;
  return VK_SUCCESS;
}
