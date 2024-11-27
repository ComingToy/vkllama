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
                const int axis, Tensor::DType const dtype)
    : Op (gpu, command), num_ (num), axis_ (axis), dtype_ (dtype)
{
  axis_ = axis_ < 0 ? 2 : axis_;
}

absl::Status
Concat::init () noexcept
{
  if (axis_ > 2 || (dtype_ == FP16 && !dev_->support_16bit_storage ()))
    {
      return absl::InvalidArgumentError (
          "fp16 dtype is unsupported on device");
    }

  const uint8_t *spv_code = nullptr;
  size_t spv_size = 0;

  if (dtype_ == FP16)
    {
      spv_code = __get_concat_fp16_comp_spv_code ();
      spv_size = __get_concat_fp16_comp_spv_size ();
    }
  else
    {
      return absl::InvalidArgumentError (
          "Concat op: only fp16 dtype is supported.");
    }

  std::vector<std::unique_ptr<Pipeline> > pipelines;

  for (int i = 0; i < num_; ++i)
    {
      Pipeline::ShaderInfo info
          = { 0, 2, sizeof (ShapeConstant) * 2 + sizeof (uint32_t), 16, 2, 1 };
      ShaderConstants specs;

      auto pipeline
          = std::make_unique<Pipeline> (dev_, spv_code, spv_size, specs, info);

      auto ret = pipeline->init ();
      if (!ret.ok ())
        {
          return ret;
        }
      pipelines.push_back (std::move (pipeline));
    }

  pipelines_.swap (pipelines);
  return absl::OkStatus ();
}

absl::StatusOr<Tensor>
Concat::operator() (const std::vector<Tensor> &inputs) noexcept
{
  if (inputs.size () != num_)
    {
      return absl::InvalidArgumentError (absl::StrFormat (
          "operator has %zu inputs but %zu are given", num_, inputs.size ()));
    }

  if (std::any_of (inputs.cbegin (), inputs.end (),
                   [this] (auto const &t) { return t.dtype () != dtype_; }))
    {
      return absl::InvalidArgumentError ("dtype of inputs are unsupported.");
    }

  if (axis_ == 0
      && std::any_of (inputs.cbegin (), inputs.cend (),
                      [&inputs] (const Tensor &item) {
                        return item.height () != inputs.front ().height ()
                               || item.width () != inputs.front ().width ();
                      }))
    {
      return absl::InvalidArgumentError ("inputs shape error on axis 0");
    }

  if (axis_ == 1
      && std::any_of (inputs.cbegin (), inputs.cend (),
                      [&inputs] (const auto &item) {
                        return inputs.front ().channels () != item.channels ()
                               || inputs.front ().width () != item.width ();
                      }))
    {
      return absl::InvalidArgumentError ("inputs shape error on axis 1");
    }

  if (axis_ == 2
      && std::any_of (inputs.cbegin (), inputs.cend (),
                      [&inputs] (const auto &item) {
                        return inputs.front ().channels () != item.channels ()
                               || inputs.front ().height () != item.height ();
                      }))
    {
      return absl::InvalidArgumentError ("inputs shape error on axis 2");
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

  if (out_.channels () != c || out_.height () != h || out_.width () != w)
    {
      out_ = Tensor (c, h, w, dev_, dtype_);
      VKLLAMA_STATUS_OK (out_.create ());
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
          return absl::InvalidArgumentError (
              absl::StrFormat ("axis %d is unsupported.", axis_));
        }
    }

  for (int i = 0; i < num_; ++i)
    {
      const auto &inp = inputs[i];
      ShaderConstants shape = inp.shape_constant () + out_.shape_constant ();
      shape.push_back (offsets[i]);

      uint32_t group_x = (inp.width () + 1) / 2,
               group_y = (inp.height () + 1) / 2, group_z = inp.channels ();
      auto ret = pipelines_[i]->set_group (group_x, group_y, group_z);
      if (!ret.ok ())
        {
          return ret;
        }
      ret = command_->record_pipeline (*pipelines_[i], { inp, out_ }, shape);
      if (!ret.ok ())
        {
          return ret;
        }
    }

  out_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return out_;
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

