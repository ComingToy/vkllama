#include "src/ops//rope.h"
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
  freqc_host_.resize (maxlen_ * dim_ / 2);
  freqs_host_.resize (maxlen_ * dim_ / 2);

  freqc_fp16_host_.resize (maxlen_ * dim_ / 2);
  freqs_fp16_host_.resize (maxlen_ * dim_ / 2);
}

void
Rope::precompute_freq_ ()
{
  std::vector<float> freq;
  std::generate_n (
      std::back_inserter (freq), dim_ / 2, [this, n = 0] () mutable {
        float dim = dim_;
        float f = 1.0f / std::pow (10000.0f, static_cast<float> (n) / dim);
        n += 2;
        return f;
      });

  // [seqlen, headim]
  for (int i = 0; i < maxlen_; ++i)
    {
      for (int k = 0; k < dim_ / 2; ++k)
        {
          auto f = freq[k] * static_cast<float> (i);
          auto freqc = std::cos (f);
          auto freqs = std::sin (f);
          auto pos = i * dim_ / 2 + k;
          freqs_host_[pos] = freqs;
          freqs_fp16_host_[pos].u16 = __fp32_to_fp16 (freqs);
          freqc_host_[pos] = freqc;
          freqc_fp16_host_[pos].u16 = __fp32_to_fp16 (freqc);
        }
    }
}

VkResult
Rope::init () noexcept
{
  precompute_freq_ ();
  freqc_ = VkTensor (1, maxlen_, dim_, dev_, dtype_);
  freqs_ = VkTensor (1, maxlen_, dim_, dev_, dtype_);

  VkResult ret = VK_SUCCESS;
  if ((ret = freqc_.create ()) != VK_SUCCESS
      || (ret = freqs_.create ()) != VK_SUCCESS)
    {
      return ret;
    }

  if (dtype_ == VkTensor::FP32)
    {
      ret = command_->upload (freqc_host_.data (), freqc_host_.size (),
                              freqc_);
      if (ret != VK_SUCCESS)
        {
          return ret;
        }

      ret = command_->upload (freqs_host_.data (), freqs_host_.size (),
                              freqs_);
      if (ret != VK_SUCCESS)
        {
          return ret;
        }
    }
  else
    {
      ret = command_->upload (freqc_fp16_host_.data (),
                              freqc_fp16_host_.size (), freqc_);
      if (ret != VK_SUCCESS)
        {
          return ret;
        }

      ret = command_->upload (freqs_fp16_host_.data (),
                              freqs_fp16_host_.size (), freqs_);
      if (ret != VK_SUCCESS)
        {
          return ret;
        }
    }

  Pipeline::ShaderInfo shader_info_k
      = { 0, 4, 3 * sizeof (uint32_t), 16, 16, 1 };
  Pipeline::ShaderInfo shader_info_q
      = { 0, 4, 3 * sizeof (uint32_t), 16, 16, 1 };

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
                  VkTensor &out_key) noexcept
{
  if (query.width () != key.width () || query.height () != key.height ()
      || query.channels () != key.channels () || query.width () != dim_
      || query.height () > maxlen_ || query.dtype () != dtype_
      || key.dtype () != dtype_)
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

  ret = pipeline_k_->set_group (groupx, groupy, groupz);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  ret = pipeline_q_->set_group (groupx, groupy, groupz);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  ShaderConstants shape
      = { (uint32_t)query.channels (), (uint32_t)query.height (),
          (uint32_t)query.width () };
  ret = command_->record_pipeline (
      *pipeline_q_, { query, freqc_, freqs_, out_query }, shape);
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  ret = command_->record_pipeline (*pipeline_k_,
                                   { key, freqc_, freqs_, out_key }, shape);
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

const std::vector<float> &
Rope::freqc ()
{
  return freqc_host_;
}

const std::vector<float> &
Rope::freqs ()
{
  return freqs_host_;
}
}

