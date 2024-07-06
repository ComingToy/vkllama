#include "rms_norm_v2.h"
#include "src/core/command.h"
#include "src/core/pipeline.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <cstdio>
#include <memory>
#include <vector>

namespace vkllama
{
RMSNormV2::RMSNormV2 (GPUDevice *dev, Command *command, VkTensor weight,
                      const float eps, const VkTensor::DType dtype)
    : Op (dev, command), weight_ (weight), dtype_ (dtype)
{
  Pipeline::ShaderInfo info0 = { 1, 4, 3 * sizeof (uint32_t), 64, 4, 1 };
  Pipeline::ShaderInfo info1 = { 1, 2, 3 * sizeof (uint32_t), 1, 128, 1 };
  Pipeline::ShaderInfo info2 = { 0, 3, 3 * sizeof (uint32_t), 16, 16, 1 };

  const auto *spv_code0 = __get_rms_norm_stage0_fp16a_comp_spv_code ();
  auto spv_size0 = __get_rms_norm_stage0_fp16a_comp_spv_size ();

  const auto *spv_code1 = __get_rms_norm_stage1_fp16a_comp_spv_code ();
  auto spv_size1 = __get_rms_norm_stage1_fp16a_comp_spv_size ();

  const auto *spv_code2 = __get_rms_norm_stage2_fp16a_comp_spv_code ();
  auto spv_size2 = __get_rms_norm_stage2_fp16a_comp_spv_size ();

  pipeline0_.reset (new Pipeline (dev_, spv_code0, spv_size0,
                                  { __fp32_to_fp16 (2.0f) }, info0));
  pipeline1_.reset (new Pipeline (dev_, spv_code1, spv_size1,
                                  { __fp32_to_fp16 (eps) }, info1));

  pipeline2_.reset (new Pipeline (dev_, spv_code2, spv_size2, {}, info2));
}

VkResult
RMSNormV2::init () noexcept
{
  if (weight_.dtype () != dtype_)
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  if (auto ret = pipeline0_->init (); ret != VK_SUCCESS)
    {
      return ret;
    }

  if (auto ret = pipeline1_->init (); ret != VK_SUCCESS)
    {
      return ret;
    }

  if (auto ret = pipeline2_->init (); ret != VK_SUCCESS)
    {
      return ret;
    }

  if (auto ret = pipeline0_->update_bindings ({ weight_ }, { 1 });
      ret != VK_SUCCESS)
    {
      return ret;
    }

  return VK_SUCCESS;
}

VkResult
RMSNormV2::operator() (VkTensor x, VkTensor &y) noexcept
{
  if (x.dtype () != dtype_)
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  // stage0
  const auto &info0 = pipeline0_->shader_info ();
  uint32_t group_x = (x.width () + info0.local_x - 1) / info0.local_x;
  uint32_t group_y = (x.height () + info0.local_y - 1) / info0.local_y;
  uint32_t group_z = (x.channels () + info0.local_z - 1) / info0.local_z;

  if (auto ret = pipeline0_->set_group (group_x, group_y, group_z);
      ret != VK_SUCCESS)
    {
      return ret;
    }

  stage0_out1_ = VkTensor::like (x);
  if (auto ret = stage0_out1_.create (); ret != VK_SUCCESS)
    {
      return ret;
    }

  stage0_out0_
      = VkTensor (x.channels (), x.height (), group_x, dev_, x.dtype ());
  if (auto ret = stage0_out0_.create (); ret != VK_SUCCESS)
    {
      return ret;
    }

  ShaderConstants constants0 = { (uint32_t)x.channels (),
                                 (uint32_t)x.height (), (uint32_t)x.width () };
  auto ret = command_->record_pipeline (
      *pipeline0_, { x, stage0_out0_, stage0_out1_ }, { 0, 2, 3 }, constants0);

  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  stage0_out0_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  stage0_out0_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  stage0_out1_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  stage0_out1_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // stage1
  const auto &info1 = pipeline1_->shader_info ();
  stage1_out0_ = VkTensor (x.channels (), x.height (), 1, dev_, x.dtype ());
  group_x = (stage1_out0_.width () + info1.local_x - 1) / info1.local_x;
  group_y = (stage1_out0_.height () + info1.local_y - 1) / info1.local_y;
  group_z = (stage1_out0_.channels () + info1.local_z - 1) / info1.local_z;

  if (auto ret = pipeline1_->set_group (group_x, group_y, group_z);
      ret != VK_SUCCESS)
    {
      return ret;
    }

  if (auto ret = stage1_out0_.create (); ret != VK_SUCCESS)
    {
      return ret;
    }

  ret = command_->record_pipeline (*pipeline1_, { stage0_out0_, stage1_out0_ },
                                   { (uint32_t)stage0_out0_.channels (),
                                     (uint32_t)stage0_out0_.height (),
                                     (uint32_t)stage0_out0_.width () });
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  stage1_out0_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  stage1_out0_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // stage2
  y = VkTensor::like (x);
  if (auto ret = y.create (); ret != VK_SUCCESS)
    {
      return ret;
    }

  const auto info2 = pipeline2_->shader_info ();
  group_x = (y.width () + info2.local_x - 1) / info2.local_x;
  group_y = (y.height () + info2.local_y - 1) / info2.local_y;
  group_z = (y.channels () + info2.local_z - 1) / info2.local_z;

  if (auto ret = pipeline2_->set_group (group_x, group_y, group_z);
      ret != VK_SUCCESS)
    {
      return ret;
    }

  fprintf (stderr,
           "shape of output = (%zu, %zu, %zu), groups = (%u, %u, %u)\n",
           y.channels (), y.height (), y.width (), group_z, group_y, group_x);

  ret = command_->record_pipeline (
      *pipeline2_, { stage0_out1_, stage1_out0_, y },
      { (uint32_t)y.channels (), (uint32_t)y.height (),
        (uint32_t)y.width () });

  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  y.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  y.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

#if 0
  auto stage00_buf = std::make_shared<std::vector<__vkllama_fp16_t> > (
      stage0_out0_.size ());
  auto stage01_buf = std::make_shared<std::vector<__vkllama_fp16_t> > (
      stage0_out1_.size ());
  auto stage10_buf = std::make_shared<std::vector<__vkllama_fp16_t> > (
      stage1_out0_.size ());
  auto out_buf = std::make_shared<std::vector<__vkllama_fp16_t> > (y.size ());

  ret = command_->download (stage0_out0_, stage00_buf->data (),
                            stage00_buf->size ());
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  ret = command_->download (stage0_out1_, stage01_buf->data (),
                            stage01_buf->size ());
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  ret = command_->download (stage1_out0_, stage10_buf->data (),
                            stage10_buf->size ());
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  ret = command_->download (y, out_buf->data (), out_buf->size ());
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  auto print_op
      = [] (const char *name, std::vector<__vkllama_fp16_t> const &buf) {
          fprintf (stderr, "%s : ", name);
          for (auto v : buf)
            {
              fprintf (stderr, "%f ", __fp16_to_fp32 (v.u16));
            }
          fprintf (stderr, "\n");
        };
  command_->defer (
      [stage00_buf, stage01_buf, stage10_buf, out_buf, print_op] () {
        print_op ("stage0_out0", *stage00_buf);
        print_op ("stage0_out1", *stage01_buf);
        print_op ("stage1_out0", *stage10_buf);
        print_op ("output", *out_buf);
        return VK_SUCCESS;
      });
#endif
  return VK_SUCCESS;
}

uint64_t
RMSNormV2::time () noexcept
{
  return pipeline0_->time () + pipeline1_->time () + pipeline2_->time ();
}
}
