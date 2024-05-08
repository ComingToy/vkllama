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

Concat::Concat (GPUDevice *gpu, Command *command, const int num)
    : Op (gpu, command), num_ (num)
{
}

VkResult
Concat::init () noexcept
{
  std::vector<std::unique_ptr<Pipeline> > pipelines;
  for (int i = 0; i < num_; ++i)
    {
      Pipeline::ShaderInfo info = { 0, 2, 5, 16, 16, 1 };
      std::vector<Pipeline::ConstantType> specs;
      auto pipeline = std::make_unique<Pipeline> (
          dev_, __get_concat_axis2_comp_spv_code (),
          __get_concat_axis2_comp_spv_size (), specs, info);

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

  auto const &ref = inputs[0];
  size_t w = 0;
  for (auto &inp : inputs)
    {
      if (inp.channels () != ref.channels () || inp.height () != ref.height ())
        {
          return VK_ERROR_UNKNOWN;
        }

      w += inp.width ();
    }

  output = VkTensor (ref.channels (), ref.height (), w, dev_);
  auto ret = output.create ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  uint32_t offset = 0;
  for (int i = 0; i < num_; ++i)
    {
      const auto &inp = inputs[i];
      std::vector<Pipeline::ConstantType> constants
          = { { .u32 = (uint32_t)inp.channels () },
              { .u32 = (uint32_t)inp.height () },
              { .u32 = (uint32_t)inp.width () },
              { .u32 = (uint32_t)w },
              { .u32 = offset } };
      uint32_t group_x = (inp.width () + 15) / 16,
               group_y = (inp.height () + 15) / 16, group_z = inp.channels ();
      pipelines_[i]->set_group (group_x, group_y, group_z);
      ret = command_->record_pipeline (*pipelines_[i], { inp, output },
                                       constants);
      if (ret != VK_SUCCESS)
        {
          return ret;
        }

      offset += inp.width ();
    }

  output.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  output.set_pipeline_stage (VK_SHADER_STAGE_COMPUTE_BIT);
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