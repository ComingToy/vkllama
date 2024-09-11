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
  auto input0 = random_tensor<Eigen::half> (gpu_, command_, params.C, params.H,
                                            params.W);

  Transpose transpose_op (gpu_, command_, 0, (Tensor::DType)params.dtype);
  ASSERT_EQ (transpose_op.init (), absl::OkStatus ());

  absl::StatusOr<Tensor> output;
  ASSERT_TRUE ((output = transpose_op (input0->first)).ok ())
      << output.status ();

  std::vector<Eigen::half> output_buf (output->size ());

  ASSERT_EQ (
      command_->download (*output, output_buf.data (), output_buf.size ()),
      absl::OkStatus ());
  ASSERT_EQ (command_->end (), absl::OkStatus ());
  ASSERT_EQ (command_->submit_and_wait (), absl::OkStatus ());

  _Tensor<Eigen::half, 3> input_tensor = _TensorMap<Eigen::half, 3> (
      input0->second.data (), (Eigen::Index)params.C, (Eigen::Index)params.H,
      (Eigen::Index)params.W);

  Eigen::array<Eigen::Index, 3> shuffle = { 1, 0, 2 };

  _Tensor<Eigen::half, 3> vk_output_tensor = _TensorMap<Eigen::half, 3> (
      output_buf.data (), (Eigen::Index)output->channels (),
      (Eigen::Index)output->height (), (Eigen::Index)output->width ());

  _Tensor<Eigen::half, 3> err (vk_output_tensor.dimensions ());
  err.setConstant (Eigen::half (params.dtype ? 1e-2 : 1e-3));

  _Tensor<Eigen::half, 3> transposed = input_tensor.shuffle (shuffle);

  _Tensor<int, 0> diff
      = ((vk_output_tensor - transposed).abs () > err).cast<int> ().sum ();

  ASSERT_EQ (*diff.data (), 0);
}

std::vector<TestTransposeParams> params = {
  { 1, 128, 64, 1 }, { 63, 53, 33, 1 }, { 32, 15, 100, 1 }, { 15, 32, 100, 1 }
};

INSTANTIATE_TEST_SUITE_P (test_transpose, TestTranspose,
                          ::testing::ValuesIn (params));
}
