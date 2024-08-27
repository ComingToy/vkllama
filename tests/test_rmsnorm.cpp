#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/cast.h"
#include "ops/rms_norm.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>

namespace vkllama
{
struct TestRMSNormParams
{
  const int C;
  const int H;
  const int W;
  const int dtype;
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

  VkTensor input0_fp16, input0_fp32, input1_fp16, input1_fp32;
  std::vector<float> input0_buf, input1_buf;

  Cast cast_input0_fp16 (gpu_, command_, VkTensor::FP32, VkTensor::FP16),
      cast_input0_fp32 (gpu_, command_, VkTensor::FP16, VkTensor::FP32),
      cast_input1_fp16 (gpu_, command_, VkTensor::FP32, VkTensor::FP16),
      cast_input1_fp32 (gpu_, command_, VkTensor::FP16, VkTensor::FP32);
  if (params.dtype)
    {
      ASSERT_EQ (cast_input0_fp16.init (), VK_SUCCESS);
      ASSERT_EQ (cast_input0_fp32.init (), VK_SUCCESS);
      ASSERT_EQ (cast_input1_fp16.init (), VK_SUCCESS);
      ASSERT_EQ (cast_input1_fp32.init (), VK_SUCCESS);

      ASSERT_EQ (cast_input0_fp16 (input0->first, input0_fp16), VK_SUCCESS);
      ASSERT_EQ (cast_input0_fp32 (input0_fp16, input0_fp32), VK_SUCCESS);
      ASSERT_EQ (cast_input1_fp16 (input1->first, input1_fp16), VK_SUCCESS);
      ASSERT_EQ (cast_input1_fp32 (input1_fp16, input1_fp32), VK_SUCCESS);

      input0_buf.resize (input0_fp32.size ());
      input1_buf.resize (input1_fp32.size ());
      ASSERT_EQ (command_->download (input0_fp32, input0_buf.data (),
                                     input0_buf.size ()),
                 VK_SUCCESS);

      ASSERT_EQ (command_->download (input1_fp32, input1_buf.data (),
                                     input1_buf.size ()),
                 VK_SUCCESS);
    }
  else
    {
      input0_fp32 = input0->first;
      input1_fp32 = input1->first;
      input0_buf.swap (input0->second);
      input1_buf.swap (input1->second);
    }

  RMSNorm norm_op (gpu_, command_, params.dtype ? input1_fp16 : input1_fp32,
                   1e-3f, (VkTensor::DType)params.dtype);
  ASSERT_EQ (norm_op.init (), VK_SUCCESS);

  VkTensor output;
  ASSERT_EQ (norm_op (params.dtype ? input0_fp16 : input0_fp32, output),
             VK_SUCCESS);

  VkTensor output_fp32;
  Cast cast_output_op (gpu_, command_, VkTensor::FP16, VkTensor::FP32);
  if (params.dtype)
    {
      ASSERT_EQ (cast_output_op.init (), VK_SUCCESS);
      ASSERT_EQ (cast_output_op (output, output_fp32), VK_SUCCESS);
    }
  else
    {
      output_fp32 = output;
    }

  std::vector<float> output_buf (output.size ());
  std::cerr << "size of output buf: " << output_buf.size () << std::endl;

  ASSERT_EQ (
      command_->download (output_fp32, output_buf.data (), output_buf.size ()),
      VK_SUCCESS);

  ASSERT_EQ (command_->end (), VK_SUCCESS) << "failed at end commands";
  ASSERT_EQ (command_->submit_and_wait (), VK_SUCCESS)
      << "failed at submit commands";

  Tensor<3> vk_output_tensor = TensorMap<3> (
      output_buf.data (), (Eigen::Index)output.channels (),
      (Eigen::Index)output.height (), (Eigen::Index)output.width ());

  Tensor<3> input_tensor0 = TensorMap<3> (
      input0_buf.data (), (Eigen::Index)input0->first.channels (),
      (Eigen::Index)input0->first.height (),
      (Eigen::Index)input0->first.width ());

  Tensor<3> input_tensor1 = TensorMap<3> (
      input1_buf.data (), (Eigen::Index)input1->first.channels (),
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
  // std::cerr << "input tensor: " << input_tensor0 << std::endl
  //           << "vk output tensor: " << vk_output_tensor << std::endl
  //           << "eigen output tensor: " << eigen_output_tensor << std::endl;
  Tensor<3> err (vk_output_tensor.dimensions ());
  err.setConstant (1e-3);
  _Tensor<int, 0> diff
      = ((vk_output_tensor - eigen_output_tensor).abs () > err)
            .cast<int> ()
            .sum ();
  ASSERT_EQ (*diff.data (), 0);
};

std::vector<TestRMSNormParams> params = {
  { 3, 2, 133, 1 }, { 3, 1023, 63, 1 },
  // { 1, 1023, 63, 0 },
  // { 3, 1023, 63, 0 }
};

INSTANTIATE_TEST_SUITE_P (test_rmsnorm, TestRMSNorm,
                          ::testing::ValuesIn (params));
}
