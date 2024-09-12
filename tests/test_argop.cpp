#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/argop.h"
#include "ops/cast.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <memory>
#include <vector>

namespace vkllama
{
struct TestArgOpParams
{
  const uint32_t C;
  const uint32_t H;
  const uint32_t W;
  const int op_type;
  const int dtype; // 0: fp32 1: fp16
};

class TestArgOp : public ::testing::TestWithParam<TestArgOpParams>
{
public:
  GPUDevice *gpu_;
  Command *command_;

  void
  SetUp () override
  {
    gpu_ = new GPUDevice ();
    command_ = new Command (gpu_);
    gpu_->init ();
    command_->init ();
  }

  void
  TearDown () override
  {
    delete command_;
    delete gpu_;
  }
};

TEST_P (TestArgOp, test_argop)
{
  ASSERT_EQ (command_->begin (), absl::OkStatus ());
  const auto params = GetParam ();
  const auto op_type = params.op_type;

  auto input0 = random_tensor<__vkllama_fp16_t> (
      gpu_, command_, params.C, params.H, params.W, __fp32_to_fp16 (-10.0f),
      __fp32_to_fp16 (10.0f));

  ASSERT_TRUE (input0);

  absl::StatusOr<Tensor> output;

  ArgMax argmax (gpu_, command_, (Tensor::DType)params.dtype);
  ArgMin argmin (gpu_, command_, (Tensor::DType)params.dtype);

  if (op_type == 0)
    {
      ASSERT_EQ (argmax.init (), absl::OkStatus ());
      output = argmax (input0->first);
      ASSERT_EQ (output.status (), absl::OkStatus ());
    }
  else if (op_type == 1)
    {
      ASSERT_EQ (argmin.init (), absl::OkStatus ());
      output = argmin (input0->first);
      ASSERT_EQ (output.status (), absl::OkStatus ());
    }

  std::vector<uint8_t> output_buf (output->size () * output->elem_bytes ());
  ASSERT_EQ (
      command_->download (*output, output_buf.data (), output_buf.size ()),
      absl::OkStatus ());
  ASSERT_EQ (command_->end (), absl::OkStatus ()) << "failed at end commands";
  ASSERT_EQ (command_->submit_and_wait (), absl::OkStatus ())
      << "failed at submit commands";

  _Tensor<uint32_t, 2> vk_output_tensor = _TensorMap<uint32_t, 2> (
      (uint32_t *)output_buf.data (), (Eigen::Index)output->channels (),
      (Eigen::Index)output->height ());

  _Tensor<Eigen::half, 3> input0_tensor
      = _TensorMap<Eigen::half, 3> ((Eigen::half *)input0->second.data (),
                                    (Eigen::Index)input0->first.channels (),
                                    (Eigen::Index)input0->first.height (),
                                    (Eigen::Index)input0->first.width ());

  _Tensor<uint32_t, 2> eigen_output_tensor;
  if (op_type == 0)
    {
      eigen_output_tensor = input0_tensor.argmax (2).cast<uint32_t> ();
    }
  else if (op_type == 1)
    {
      eigen_output_tensor = input0_tensor.argmin (2).cast<uint32_t> ();
    }

#if 0
  for (Eigen::Index c = 0; c < input0_tensor.dimension (0); ++c)
    {
      for (Eigen::Index h = 0; h < input0_tensor.dimension (1); ++h)
        {
          if (eigen_output_tensor (c, h) != vk_output_tensor (c, h))
            {
              Eigen::array<Eigen::Index, 3> start = { c, h, 0 };
              Eigen::array<Eigen::Index, 3> extents
                  = { 1, 1, input0_tensor.dimension (2) };

              std::cerr << "params.op_type: " << params.op_type
                        << "eigen output: " << eigen_output_tensor (c, h)
                        << " with value: "
                        << input0_tensor (c, h, eigen_output_tensor (c, h))
                        << ", vk output: " << vk_output_tensor (c, h)
                        << " with value: "
                        << input0_tensor (c, h, vk_output_tensor (c, h))
                        << std::endl
                        << "input tensor: "
                        << input0_tensor.slice (start, extents) << std::endl;
              break;
            }
        }
    }
#endif

  _Tensor<uint32_t, 0> diff
      = (eigen_output_tensor != vk_output_tensor).cast<uint32_t> ().sum ();
  ASSERT_LE (*diff.data (), 20);
};
#if 1
std::vector<TestArgOpParams> params = {
  { 1, 1023, 511, 0, 1 },
  { 1, 1023, 511, 1, 1 },
  { 3, 1023, 511, 0, 1 },
  { 3, 1023, 511, 1, 1 },
};
#else
std::vector<TestArgOpParams> params = { { 1, 32, 64, 0, 0 } };
#endif

INSTANTIATE_TEST_SUITE_P (test_argop, TestArgOp, ::testing::ValuesIn (params));
}
