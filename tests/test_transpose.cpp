#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/argop.h"
#include "ops/cast.h"
#include "ops/transpose.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <memory>
#include <vector>

namespace vkllama
{
struct TestTransposeParams
{
  int C;
  int H;
  int W;
  int dtype;
};

class TestTranspose : public ::testing::TestWithParam<TestTransposeParams>
{
public:
  GPUDevice *gpu_;
  Command *command_;

  void
  SetUp () override
  {
    gpu_ = new GPUDevice ();
    command_ = new Command (gpu_);
    ASSERT_EQ (gpu_->init (), absl::OkStatus ());
    ASSERT_EQ (command_->init (), absl::OkStatus ());
  }

  void
  TearDown () override
  {
    delete command_;
    delete gpu_;
  }
};

TEST_P (TestTranspose, test_transpose)
{
  ASSERT_EQ (command_->begin (), absl::OkStatus ());
  auto params = GetParam ();
  auto input0
      = random_tensor<float> (gpu_, command_, params.C, params.H, params.W);

  Tensor input_fp32, input_fp16;
  std::vector<float> input_buf;
  Cast cast_input_op0 (gpu_, command_, Tensor::FP32, Tensor::FP16);
  Cast cast_input_op1 (gpu_, command_, Tensor::FP16, Tensor::FP32);
  if (params.dtype)
    {
      ASSERT_EQ (cast_input_op0.init (), absl::OkStatus ());
      ASSERT_EQ (cast_input_op1.init (), absl::OkStatus ());
      ASSERT_EQ (cast_input_op0 (input0->first, input_fp16),
                 absl::OkStatus ());
      ASSERT_EQ (cast_input_op1 (input_fp16, input_fp32), absl::OkStatus ());

      input_buf.resize (input_fp16.size ());
      ASSERT_EQ (command_->download (input_fp32, input_buf.data (),
                                     input_buf.size ()),
                 absl::OkStatus ());
    }
  else
    {
      input_fp32 = input0->first;
      input_buf.swap (input0->second);
    }

  Transpose transpose_op (gpu_, command_, 0, (Tensor::DType)params.dtype);
  ASSERT_EQ (transpose_op.init (), absl::OkStatus ());

  Tensor output;
  if (params.dtype)
    {
      transpose_op (input_fp16, output);
    }
  else
    {
      transpose_op (input_fp32, output);
    }

  std::vector<float> output_buf (output.size ());
  Tensor output_fp32;
  Cast cast_output_op (gpu_, command_, Tensor::FP16, Tensor::FP32);
  if (params.dtype)
    {
      ASSERT_EQ (cast_output_op.init (), absl::OkStatus ());
      ASSERT_EQ (cast_output_op (output, output_fp32), absl::OkStatus ());
    }
  else
    {
      output_fp32 = output;
    }

  ASSERT_EQ (
      command_->download (output_fp32, output_buf.data (), output_buf.size ()),
      absl::OkStatus ());
  ASSERT_EQ (command_->end (), absl::OkStatus ());
  ASSERT_EQ (command_->submit_and_wait (), absl::OkStatus ());

  Tensor<3> input_tensor
      = TensorMap<3> (input_buf.data (), (Eigen::Index)params.C,
                      (Eigen::Index)params.H, (Eigen::Index)params.W);

  Eigen::array<Eigen::Index, 3> shuffle = { 1, 0, 2 };

  Tensor<3> vk_output_tensor = TensorMap<3> (
      output_buf.data (), (Eigen::Index)output_fp32.channels (),
      (Eigen::Index)output_fp32.height (), (Eigen::Index)output_fp32.width ());

  Tensor<3> err (vk_output_tensor.dimensions ());
  err.setConstant (params.dtype ? 1e-2 : 1e-3);

  if (params.dtype)
    {
      auto transposed
          = input_tensor.cast<Eigen::half> ().shuffle (shuffle).cast<float> ();
      _Tensor<int, 0> diff
          = ((vk_output_tensor - transposed).abs () > err).cast<int> ().sum ();

      ASSERT_EQ (*diff.data (), 0);
    }
  else
    {
      auto transposed = input_tensor.shuffle (shuffle);
      _Tensor<int, 0> diff
          = ((vk_output_tensor - transposed).abs () > err).cast<int> ().sum ();

      ASSERT_EQ (*diff.data (), 0);
    }
}

std::vector<TestTransposeParams> params
    = { { 1, 128, 64, 0 },  { 63, 53, 33, 0 }, { 32, 15, 100, 0 },
        { 15, 32, 100, 0 }, { 1, 128, 64, 1 }, { 63, 53, 33, 1 },
        { 32, 15, 100, 1 }, { 15, 32, 100, 1 } };

INSTANTIATE_TEST_SUITE_P (test_transpose, TestTranspose,
                          ::testing::ValuesIn (params));
}
