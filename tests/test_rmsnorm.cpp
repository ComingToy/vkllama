#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/rms_norm.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>

namespace
{
struct TestRMSNormParams
{
  const int C;
  const int H;
  const int W;
};

using TestRMSNorm = VkllamaTestWithParam<TestRMSNormParams>;

TEST_P (TestRMSNorm, test_rmsnorm)
{
  ASSERT_EQ (command_->begin (), VK_SUCCESS) << "failed at begin commands";

  auto params = GetParam ();
  auto input0
      = random_tensor<float> (gpu_, command_, params.C, params.H, params.W);
  auto input1 = random_tensor<float> (gpu_, command_, 1, 1, params.W);

  ASSERT_TRUE (input0);
  ASSERT_TRUE (input1);

  RMSNorm norm_op (gpu_, command_, input1->first);
  ASSERT_EQ (norm_op.init (), VK_SUCCESS);

  VkTensor output;
  ASSERT_EQ (norm_op (input0->first, output), VK_SUCCESS);
  std::vector<float> output_buf (output.size ());

  ASSERT_EQ (
      command_->download (output, output_buf.data (), output_buf.size ()),
      VK_SUCCESS);

  ASSERT_EQ (command_->end (), VK_SUCCESS) << "failed at end commands";
  ASSERT_EQ (command_->submit_and_wait (), VK_SUCCESS)
      << "failed at submit commands";

  Tensor<3> vk_output_tensor = TensorMap<3> (
      output_buf.data (), (Eigen::Index)output.channels (),
      (Eigen::Index)output.height (), (Eigen::Index)output.width ());

  Tensor<3> input_tensor0 = TensorMap<3> (
      input0->second.data (), (Eigen::Index)input0->first.channels (),
      (Eigen::Index)input0->first.height (),
      (Eigen::Index)input0->first.width ());

  Tensor<3> input_tensor1 = TensorMap<3> (
      input1->second.data (), (Eigen::Index)input1->first.channels (),
      (Eigen::Index)input1->first.height (),
      (Eigen::Index)input1->first.width ());

  Eigen::array<Eigen::Index, 1> mean_dims = { 2 };
  Eigen::array<Eigen::Index, 3> dims
      = { input_tensor0.dimension (0), input_tensor0.dimension (1), 1 };
  Eigen::array<Eigen::Index, 3> broadcasts
      = { 1, 1, input_tensor0.dimension (2) };
  Eigen::array<Eigen::Index, 3> weight_broadcasts
      = { input_tensor0.dimension (0), input_tensor0.dimension (1), 1 };

  Tensor<3> eigen_output_tensor
      = (input_tensor0.pow (2.0f).mean (mean_dims) + 1e-3f)
            .rsqrt ()
            .reshape (dims)
            .broadcast (broadcasts)
        * input_tensor1.broadcast (weight_broadcasts) * input_tensor0;
  // std::cerr << "eigen output mean: " << eigen_output_tensor.mean ()
  //           << std::endl
  //           << "vk output mean: " << vk_output_tensor.mean () << std::endl;

  Tensor<3> err (vk_output_tensor.dimensions ());
  err.setConstant (1e-3);
  _Tensor<int, 0> diff
      = ((vk_output_tensor - eigen_output_tensor).abs () > err)
            .cast<int> ()
            .sum ();
  ASSERT_EQ (*diff.data (), 0);
};

std::vector<TestRMSNormParams> params = { { 1, 1023, 63 }, { 3, 1023, 63 } };
INSTANTIATE_TEST_SUITE_P (test_rmsnorm, TestRMSNorm,
                          ::testing::ValuesIn (params));
}
