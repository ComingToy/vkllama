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
    ASSERT_EQ (gpu_->init (), VK_SUCCESS);
    ASSERT_EQ (command_->init (), VK_SUCCESS);
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
  ASSERT_EQ (command_->begin (), VK_SUCCESS);

  auto input0 = random_tensor<float> (gpu_, command_, params.shape.C,
                                      params.shape.H, params.shape.W);
  ASSERT_TRUE (input0);
  VkTensor input_tensor;

  Cast cast (gpu_, command_, VkTensor::FP32, VkTensor::FP16);
  if (params.dtype)
    {
      ASSERT_EQ (cast.init (), VK_SUCCESS);
      ASSERT_EQ (cast (input0->first, input_tensor), VK_SUCCESS);
    }
  else
    {
      input_tensor = input0->first;
    }

  Slice slice_op (gpu_, command_, (VkTensor::DType)params.dtype);
  ASSERT_EQ (slice_op.init (), VK_SUCCESS);

  VkTensor out;
  ASSERT_EQ (slice_op (input_tensor, params.starts, params.extents, out),
             VK_SUCCESS);

  VkTensor out_fp32;
  Cast cast_output_op (gpu_, command_, VkTensor::FP16, VkTensor::FP32);

  if (params.dtype)
    {
      ASSERT_EQ (cast_output_op.init (), VK_SUCCESS);
      ASSERT_EQ (cast_output_op (out, out_fp32), VK_SUCCESS);
    }
  else
    {
      out_fp32 = out;
    }

  std::vector<float> output_buf (out_fp32.size ());

  ASSERT_EQ (
      command_->download (out_fp32, output_buf.data (), output_buf.size ()),
      VK_SUCCESS);

  ASSERT_EQ (command_->end (), VK_SUCCESS);
  ASSERT_EQ (command_->submit (), VK_SUCCESS);
  ASSERT_EQ (command_->wait (), VK_SUCCESS);

  Tensor<3> vk_output_tensor
      = TensorMap<3> (output_buf.data (), out_fp32.channels (),
                      out_fp32.height (), out_fp32.width ());

  Tensor<3> input_eigen_tensor = TensorMap<3> (
      input0->second.data (), (Eigen::Index)params.shape.C,
      (Eigen::Index)params.shape.H, (Eigen::Index)params.shape.W);

  Eigen::array<Eigen::Index, 3> starts
      = { params.starts[0], params.starts[1], params.starts[2] };
  Eigen::array<Eigen::Index, 3> extents
      = { params.extents[0], params.extents[1], params.extents[2] };

  Tensor<3> output_tensor;
  if (params.dtype)
    {
      output_tensor = input_eigen_tensor.cast<Eigen::half> ()
                          .slice (starts, extents)
                          .cast<float> ();
    }
  else
    {
      output_tensor = input_eigen_tensor.slice (starts, extents);
    }

  Tensor<3> err (vk_output_tensor.dimensions ());
  err.setConstant (params.dtype ? 1e-2 : 1e-3);
  _Tensor<int, 0> diff
      = ((vk_output_tensor - output_tensor).abs () > err).cast<int> ().sum ();

  // std::cerr << "input tensor: " << input_eigen_tensor << std::endl
  //           << "output tensor: " << output_tensor << std::endl
  //           << "vk output tensor: " << vk_output_tensor << std::endl;
  ASSERT_EQ (*diff.data (), 0);
}

std::vector<TestSliceParams> params = {
  // { 0, { 1, 65, 33 }, { 0, 0, 0 }, { 1, 33, 22 } },
  // { 0, { 3, 65, 33 }, { 1, 5, 8 }, { 1, 33, 22 } },
  { 1, { 32, 1024, 100 }, { 0, 0, 0 }, { 32, 33, 100 } },
  // { 1, { 1, 65, 33 }, { 0, 0, 0 }, { 1, 33, 22 } },
  // { 1, { 3, 65, 33 }, { 1, 5, 8 }, { 1, 33, 22 } },
};

INSTANTIATE_TEST_SUITE_P (test_slice, TestSlice, ::testing::ValuesIn (params));
}
