#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/cast.h"
#include "ops/slice.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <array>
#include <memory>
#include <tuple>
#include <vector>

namespace vkllama
{
struct TestSliceParams
{
  int dtype;
  struct
  {
    int C, H, W;
  } shape;

  std::array<uint32_t, 3> starts, extents;
};

class TestSlice : public ::testing::TestWithParam<TestSliceParams>
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

TEST_P (TestSlice, test_slice)
{
  auto params = GetParam ();
  ASSERT_EQ (command_->begin (), absl::OkStatus ());

  auto input0 = random_tensor<Eigen::half> (gpu_, command_, params.shape.C,
                                            params.shape.H, params.shape.W);
  ASSERT_TRUE (input0);

  Slice slice_op (gpu_, command_, (Tensor::DType)params.dtype);
  ASSERT_EQ (slice_op.init (), absl::OkStatus ());

  absl::StatusOr<Tensor> out;
  ASSERT_EQ ((out = slice_op (input0->first, params.starts, params.extents))
                 .status (),
             absl::OkStatus ());

  std::vector<Eigen::half> output_buf (out->size ());

  ASSERT_EQ (command_->download (*out, output_buf.data (), output_buf.size ()),
             absl::OkStatus ());

  ASSERT_EQ (command_->end (), absl::OkStatus ());
  ASSERT_EQ (command_->submit (), absl::OkStatus ());
  ASSERT_EQ (command_->wait (), absl::OkStatus ());

  _Tensor<Eigen::half, 3> vk_output_tensor = _TensorMap<Eigen::half, 3> (
      output_buf.data (), (Eigen::Index)out->channels (),
      (Eigen::Index)out->height (), (Eigen::Index)out->width ());

  _Tensor<Eigen::half, 3> input_eigen_tensor = _TensorMap<Eigen::half, 3> (
      input0->second.data (), (Eigen::Index)params.shape.C,
      (Eigen::Index)params.shape.H, (Eigen::Index)params.shape.W);

  Eigen::array<Eigen::Index, 3> starts
      = { params.starts[0], params.starts[1], params.starts[2] };
  Eigen::array<Eigen::Index, 3> extents
      = { params.extents[0], params.extents[1], params.extents[2] };

  _Tensor<Eigen::half, 3> output_tensor;
  output_tensor
      = input_eigen_tensor.cast<Eigen::half> ().slice (starts, extents);

  _Tensor<Eigen::half, 3> err (vk_output_tensor.dimensions ());
  err.setConstant (Eigen::half (params.dtype ? 1e-2 : 1e-3));

  _Tensor<int, 0> diff
      = ((vk_output_tensor - output_tensor).abs () > err).cast<int> ().sum ();

  // std::cerr << "input tensor: " << input_eigen_tensor << std::endl
  //           << "output tensor: " << output_tensor << std::endl
  //           << "vk output tensor: " << vk_output_tensor << std::endl;
  ASSERT_EQ (*diff.data (), 0);
}

std::vector<TestSliceParams> params = {
  { 1, { 32, 1024, 100 }, { 0, 0, 0 }, { 32, 33, 100 } },
  { 1, { 1, 65, 33 }, { 0, 0, 0 }, { 1, 33, 22 } },
  { 1, { 3, 65, 33 }, { 1, 5, 8 }, { 1, 33, 22 } },
};

INSTANTIATE_TEST_SUITE_P (test_slice, TestSlice, ::testing::ValuesIn (params));
}
