#include "src/ops/softmax.h"
#include "src/core/command.h"
#include "src/core/gpu_device.h"
#include "src/core/pipeline.h"
#include "src/core/tensor.h"
#include "src/ops/reduce.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <vector>

namespace vkllama
{
Softmax::Softmax (GPUDevice *dev, Command *command, bool seq_mask,
                  const VkTensor::DType dtype)
    : Op (dev, command), seq_mask_ (seq_mask), dtype_ (dtype)
{
}

VkResult
Softmax::init () noexcept
{
  reduce_.reset (new Reduce (dev_, command_, 1, dtype_));
  auto ret = reduce_->init ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  Pipeline::ShaderInfo info0 = { 1, 4, 3 * sizeof (uint32_t), 32, 4, 1 };
  Pipeline::ShaderInfo info1 = { 0, 2, 3 * sizeof (uint32_t), 1, 32, 1 };
  Pipeline::ShaderInfo info2 = { 0, 3, 3 * sizeof (uint32_t), 32, 4, 1 };

  auto spv_code0 = dtype_ == VkTensor::FP32
                       ? __get_softmax_stage0_comp_spv_code ()
                       : __get_softmax_stage0_fp16_comp_spv_code ();
  auto spv_size0 = dtype_ == VkTensor::FP32
                       ? __get_softmax_stage0_comp_spv_size ()
                       : __get_softmax_stage0_fp16_comp_spv_size ();

  if (dtype_ == VkTensor::FP16 && dev_->support_fp16_arithmetic ())
    {
      spv_code0 = __get_softmax_stage0_fp16a_comp_spv_code ();
      spv_size0 = __get_softmax_stage0_fp16a_comp_spv_size ();
    }

  softmax0_.reset (
      new Pipeline (dev_, spv_code0, spv_size0, { (int)seq_mask_ }, info0));

  auto spv_code1 = __get_softmax_stage1_comp_spv_code ();
  auto spv_size1 = __get_softmax_stage1_comp_spv_size ();
  if (dtype_ == VkTensor::FP16 && dev_->support_fp16_arithmetic ())
    {
      spv_code1 = __get_softmax_stage1_fp16a_comp_spv_code ();
      spv_size1 = __get_softmax_stage1_fp16a_comp_spv_size ();
    }

  softmax1_.reset (new Pipeline (dev_, spv_code1, spv_size1, {}, info1));

  auto spv_code2 = dtype_ == VkTensor::FP32
                       ? __get_softmax_stage2_comp_spv_code ()
                       : __get_softmax_stage2_fp16_comp_spv_code ();

  auto spv_size2 = dtype_ == VkTensor::FP32
                       ? __get_softmax_stage2_comp_spv_size ()
                       : __get_softmax_stage2_fp16_comp_spv_size ();

  if (dtype_ == VkTensor::FP16 && dev_->support_fp16_arithmetic ())
    {
      spv_code2 = __get_softmax_stage2_fp16a_comp_spv_code ();
      spv_size2 = __get_softmax_stage2_fp16a_comp_spv_size ();
    }

  softmax2_.reset (new Pipeline (dev_, spv_code2, spv_size2, {}, info2));

  if ((ret = softmax0_->init ()) != VK_SUCCESS
      || (ret = softmax1_->init ()) != VK_SUCCESS
      || (ret = softmax2_->init ()) != VK_SUCCESS)
    {
      return ret;
    }

  return VK_SUCCESS;
}

uint64_t
Softmax::time () noexcept
{
  return reduce_->time () + softmax0_->time () + softmax1_->time ()
         + softmax2_->time ();
}

VkResult
Softmax::operator() (VkTensor a, VkTensor &b) noexcept
{
  if (a.dtype () != dtype_)
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  auto ret = reduce_->operator() (a, bias_);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

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

  exps_ = VkTensor (a.channels (), a.height (), a.width (), dev_);
  if ((ret = exps_.create ()) != VK_SUCCESS)
    {
      return ret;
    }

  ret = command_->record_pipeline (*softmax0_, { a, bias_, m_, exps_ },
                                   { (uint32_t)a.channels (),
                                     (uint32_t)a.height (),
                                     (uint32_t)a.width () });
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  m_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  m_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  exps_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  exps_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

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

  group_x = 1;
  group_y = (a.height () + 31) / 32;
  group_z = a.channels ();

  softmax1_->set_group (group_x, group_y, group_z);
  ret = command_->record_pipeline (
      *softmax1_, { m_, sum_ },
      { (uint32_t)a.channels (), (uint32_t)a.height (), group_x });
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  sum_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  sum_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  group_x = (a.width () + 31) / 32;
  group_y = (a.height () + 3) / 4;
  ret = softmax2_->set_group (group_x, group_y, group_z);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }
  ret = command_->record_pipeline (*softmax2_, { exps_, sum_, out_ },
                                   { (uint32_t)a.channels (),
                                     (uint32_t)a.height (),
                                     (uint32_t)a.width () });
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  out_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  b = out_;
  return VK_SUCCESS;
}
}

