#include "src/ops/concat.h"
#include "src/core/command.h"
#include "src/core/pipeline.h"
#include "src/core/tensor.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

namespace vkllama
{
Concat::Concat (GPUDevice *gpu, Command *command, const int num,
                const int axis, VkTensor::DType const dtype)
    : Op (gpu, command), num_ (num), axis_ (axis), dtype_ (dtype)
{
  axis_ = axis_ < 0 ? 2 : axis_;
}

VkResult
Concat::init () noexcept
{
  if (axis_ > 2
      || (dtype_ == VkTensor::FP16 && !dev_->support_16bit_storage ()))
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  const auto *spv_code = dtype_ == VkTensor::FP32
                             ? __get_concat_axis2_comp_spv_code ()
                             : __get_concat_axis2_fp16_comp_spv_code ();

  size_t spv_size = dtype_ == VkTensor::FP32
                        ? __get_concat_axis2_comp_spv_size ()
                        : __get_concat_axis2_fp16_comp_spv_size ();

  std::vector<std::unique_ptr<Pipeline> > pipelines;

  for (int i = 0; i < num_; ++i)
    {
      Pipeline::ShaderInfo info = { 0, 2, sizeof (uint32_t) * 6, 16, 16, 1 };
      ShaderConstants specs;

      auto pipeline
          = std::make_unique<Pipeline> (dev_, spv_code, spv_size, specs, info);

      auto ret = pipeline->init ();
      if (ret != VK_SUCCESS)
        {
          return ret;
        }
      pipelines.push_back (std::move (pipeline));
    }

  pipelines_.swap (pipelines);
  return VK_SUCCESS;
}

VkResult
Concat::operator() (const std::vector<VkTensor> &inputs,
                    VkTensor &output) noexcept
{
  if (inputs.size () != num_)
    {
      return VK_ERROR_UNKNOWN;
    }

  if (std::any_of (inputs.cbegin (), inputs.end (),
                   [this] (auto const &t) { return t.dtype () != dtype_; }))
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  if (axis_ == 0
      && std::any_of (inputs.cbegin (), inputs.cend (),
                      [&inputs] (const VkTensor &item) {
                        return item.height () != inputs.front ().height ()
                               || item.width () != inputs.front ().width ();
                      }))
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  if (axis_ == 1
      && std::any_of (inputs.cbegin (), inputs.cend (),
                      [&inputs] (const auto &item) {
                        return inputs.front ().channels () != item.channels ()
                               || inputs.front ().width () != item.width ();
                      }))
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  if (axis_ == 2
      && std::any_of (inputs.cbegin (), inputs.cend (),
                      [&inputs] (const auto &item) {
                        return inputs.front ().channels () != item.channels ()
                               || inputs.front ().height () != item.height ();
                      }))
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  auto const &ref = inputs[0];
  size_t c = ref.channels (), h = ref.height (), w = ref.width ();
  for (size_t i = 1; i < inputs.size (); ++i)
    {
      auto &inp = inputs[i];

      if (axis_ == 0)
        {
          c += inp.channels ();
        }
      else if (axis_ == 1)
        {
          h += inp.height ();
        }
      else
        {
          w += inp.width ();
        }
    }

  output = VkTensor (c, h, w, dev_, dtype_);
  auto ret = output.create ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  std::vector<uint32_t> offsets;
  uint32_t offset = 0;
  for (size_t i = 0; i < num_; ++i)
    {
      offsets.push_back (offset);
      if (axis_ == 0)
        {
          offset += inputs[i].size ();
        }
      else if (axis_ == 1)
        {
          offset += inputs[i].width () * inputs[i].height ();
        }
      else if (axis_ == 2)
        {
          offset += inputs[i].width ();
        }
      else
        {
          return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
    }

  for (int i = 0; i < num_; ++i)
    {
      const auto &inp = inputs[i];
      ShaderConstants constants = { (uint32_t)inp.channels (),
                                    (uint32_t)inp.height (),
                                    (uint32_t)inp.width (),
                                    (uint32_t)h,
                                    (uint32_t)w,
                                    offsets[i] };

      uint32_t group_x = (inp.width () + 15) / 16,
               group_y = (inp.height () + 15) / 16, group_z = inp.channels ();
      pipelines_[i]->set_group (group_x, group_y, group_z);
      ret = command_->record_pipeline (*pipelines_[i], { inp, output },
                                       constants);
      if (ret != VK_SUCCESS)
        {
          return ret;
        }
    }

  output.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  output.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return VK_SUCCESS;
}

uint64_t
Concat::time () noexcept
{
  std::vector<uint64_t> times;
  std::generate_n (std::back_inserter (times), num_, [this, i = 0] () mutable {
    auto time = pipelines_[i]->time ();
    i += 1;
    return time;
  });

  uint64_t time = *std::max_element (times.cbegin (), times.cend ());
  return time;
}
}

