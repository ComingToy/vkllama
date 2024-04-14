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
  Pipeline::ShaderInfo info1 = { 0, 2, 3, 1, 32, 1 };
  Pipeline::ShaderInfo info2 = { 0, 3, 3, 32, 4, 1 };

  softmax0_.reset (new Pipeline (dev_, __get_softmax_stage0_comp_spv_code (),
                                 __get_softmax_stage0_comp_spv_size (), {},
                                 info0));

  softmax1_.reset (new Pipeline (dev_, __get_softmax_stage1_comp_spv_code (),
                                 __get_softmax_stage1_comp_spv_size (), {},
                                 info1));

  softmax2_.reset (new Pipeline (dev_, __get_softmax_stage2_comp_spv_code (),
                                 __get_softmax_stage2_comp_spv_size (), {},
                                 info2));

  if ((ret = softmax0_->init ()) != VK_SUCCESS
      || (ret = softmax1_->init ()) != VK_SUCCESS
      || (ret = softmax2_->init ()) != VK_SUCCESS)
    {
      return ret;
    }

  return VK_SUCCESS;
}

uint64_t
Softmax::time ()
{
  return reduce_->time () + softmax0_->time () + softmax1_->time ()
         + softmax2_->time ();
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

  out_ = VkTensor::like (a);
  ret = out_.create ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  sum_ = VkTensor (a.channels (), a.height (), 1, dev_);
  if ((ret = sum_.create ()) != VK_SUCCESS)
    {
      return ret;
    }

  shape[2] = { .u32 = group_x };
  group_x = 1;
  group_y = (a.height () + 31) / 32;
  group_z = a.channels ();

  softmax1_->set_group (group_x, group_y, group_z);
  ret = command_->record_pipeline (*softmax1_, { m_, sum_ }, shape);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  sum_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  sum_.set_pipeline_stage (VK_SHADER_STAGE_COMPUTE_BIT);

  group_x = (a.width () + 31) / 32;
  group_y = (a.height () + 3) / 4;
  shape[2] = { .u32 = (uint32_t)a.width () };
  ret = softmax2_->set_group (group_x, group_y, group_z);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }
  ret = command_->record_pipeline (*softmax2_, { exps_, sum_, out_ }, shape);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  out_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out_.set_pipeline_stage (VK_SHADER_STAGE_COMPUTE_BIT);
  b = out_;
  return VK_SUCCESS;
}
