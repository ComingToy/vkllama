#include "src/ops/rope.h"
#include "src/core/command.h"
#include "src/core/pipeline.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>
#include <vector>

namespace vkllama
{
Rope::Rope (GPUDevice *dev, Command *command, const int maxlen, const int dim,
            const VkTensor::DType dtype)
    : Op (dev, command), maxlen_ (maxlen), dim_ (dim), dtype_ (dtype)
{
}

VkResult
Rope::init () noexcept
{
  Pipeline::ShaderInfo shader_info_k
      = { 0, 2, 4 * sizeof (uint32_t), 16, 16, 1 };
  Pipeline::ShaderInfo shader_info_q
      = { 0, 2, 4 * sizeof (uint32_t), 16, 16, 1 };

  const auto *spv_code = dtype_ == VkTensor::FP32
                             ? __get_rope_comp_spv_code ()
                             : __get_rope_fp16_comp_spv_code ();

  const auto spv_size = dtype_ == VkTensor::FP32
                            ? __get_rope_comp_spv_size ()
                            : __get_rope_fp16_comp_spv_size ();

  pipeline_k_.reset (
      new Pipeline (dev_, spv_code, spv_size, {}, shader_info_k));

  pipeline_q_.reset (
      new Pipeline (dev_, spv_code, spv_size, {}, shader_info_q));

  VkResult ret = VK_SUCCESS;
  if ((ret = pipeline_k_->init ()) != VK_SUCCESS
      || (ret = pipeline_q_->init ()) != VK_SUCCESS)
    {
      return ret;
    }

  return VK_SUCCESS;
}

uint64_t
Rope::time () noexcept
{
  return std::max (pipeline_k_->time (), pipeline_q_->time ());
}

VkResult
Rope::operator() (VkTensor query, VkTensor key, VkTensor &out_query,
                  VkTensor &out_key, const size_t offset) noexcept
{
  if (query.width () != key.width () || query.channels () != key.channels ()
      || query.width () != dim_ || query.height () > maxlen_
      || query.dtype () != dtype_ || key.dtype () != dtype_)
    {
      return VK_ERROR_UNKNOWN;
    }

  out_query = VkTensor::like (query);
  out_key = VkTensor::like (key);
  auto ret = out_query.create ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }
  if ((ret = out_key.create ()) != VK_SUCCESS)
    {
      return ret;
    }

  uint32_t groupx = (query.width () / 2 + 15) / 16,
           groupy = (query.height () + 15) / 16, groupz = query.channels ();

  ret = pipeline_q_->set_group (groupx, groupy, groupz);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  groupx = (key.width () / 2 + 15) / 16;
  groupy = (key.height () + 15) / 16;
  groupz = key.channels ();

  ret = pipeline_k_->set_group (groupx, groupy, groupz);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  ShaderConstants shape
      = { (uint32_t)query.channels (), (uint32_t)query.height (),
          (uint32_t)query.width (), (uint32_t)offset };
  ret = command_->record_pipeline (*pipeline_q_, { query, out_query }, shape);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  int key_offset = std::max (int (offset - key.height ()), 0);

  ShaderConstants key_shape
      = { (uint32_t)key.channels (), (uint32_t)key.height (),
          (uint32_t)key.width (), (uint32_t)key_offset };

  ret = command_->record_pipeline (*pipeline_k_, { key, out_key }, key_shape);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  out_query.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out_query.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  out_key.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out_key.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return VK_SUCCESS;
}

}

