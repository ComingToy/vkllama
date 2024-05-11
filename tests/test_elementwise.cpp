#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/elementwise.h"
#include "tests/test_common.h"
#include "gtest/gtest.h"
#include <vector>

namespace
{
struct TestElementwiseParams
{
  const int C;
  const int H;
  const int W;
  const int op_type;
  const bool constant_b;
};

class TestElementwise : public ::testing::TestWithParam<TestElementwiseParams>
{
public:
  void
  SetUp () override
  {
    gpu_ = new GPUDevice ();
    gpu_->init ();

    command_ = new Command (gpu_);
    command_->init ();
  }

  void
  TearDown () override
  {
    delete command_;
    delete gpu_;
  }

  GPUDevice *gpu_;
  Command *command_;
};

TEST_P (TestElementwise, test_elementwise)
{
  auto params = GetParam ();
  ASSERT_EQ (command_->begin (), VK_SUCCESS) << "failed at begining commands";
  auto input0
      = random_tensor<float> (gpu_, command_, params.C, params.H, params.W);
  auto input1
      = random_tensor<float> (gpu_, command_, params.C, params.H, params.W);
  float alpha = random_number (-2.0f, 2.0f);

  ElementWise elementwise_op (gpu_, command_, params.op_type);
  ASSERT_EQ (elementwise_op.init (), VK_SUCCESS)
      << "failed at init elementwise op";

  VkTensor out;
  if (params.constant_b)
    {
      ASSERT_EQ (elementwise_op (input0->first, alpha, out), VK_SUCCESS);
    }
  else
    {
      ASSERT_EQ (elementwise_op (input0->first, input1->first, out),
                 VK_SUCCESS);
    }

  std::vector<float> output_buf (out.size ());
  ASSERT_EQ (command_->download (out, output_buf.data (), output_buf.size ()),
             VK_SUCCESS)
      << "failed at download output tensor";
  ASSERT_EQ (command_->end (), VK_SUCCESS) << "failed at edndding commands";
  ASSERT_EQ (command_->submit_and_wait (), VK_SUCCESS)
      << "failed at submiting commands";

  Tensor<3> vk_output_tensor
      = TensorMap<3> (output_buf.data (), (Eigen::Index)out.channels (),
                      (Eigen::Index)out.height (), (Eigen::Index)out.width ());
  Tensor<3> input0_tensor = TensorMap<3> (
      input0->second.data (), (Eigen::Index)input0->first.channels (),
      (Eigen::Index)input0->first.height (),
      (Eigen::Index)input0->first.width ());

  Tensor<3> input1_tensor = TensorMap<3> (
      input1->second.data (), (Eigen::Index)input1->first.channels (),
      (Eigen::Index)input1->first.height (),
      (Eigen::Index)input1->first.width ());

  Tensor<3> output_tensor;
  if (params.op_type == 0)
    {
      if (params.constant_b)
        {
          output_tensor = input0_tensor + alpha;
        }
      else
        {
          output_tensor = input0_tensor + input1_tensor;
        }
    }
  else if (params.op_type == 1)
    {
      if (params.constant_b)
        {
          output_tensor = input0_tensor - alpha;
        }
      else
        {
          output_tensor = input0_tensor - input1_tensor;
        }
    }
  else if (params.op_type == 2)
    {
      if (params.constant_b)
        {
          output_tensor = input0_tensor * alpha;
        }
      else
        {
          output_tensor = input0_tensor * input1_tensor;
        }
    }
  else if (params.op_type == 3)
    {
      if (params.constant_b)
        {
          output_tensor = input0_tensor / alpha;
        }
      else
        {
          output_tensor = input0_tensor / input1_tensor;
        }
    }

  Tensor<0> mse = (output_tensor - vk_output_tensor).pow (2.0f).mean ();
  ASSERT_LT (*mse.data (), 1e-4);
}

std::vector<TestElementwiseParams> params
    = { { 3, 1023, 512, 0, 0 }, { 3, 1023, 511, 1, 0 }, { 3, 1023, 511, 2, 0 },
        { 3, 1023, 511, 3, 0 }, { 3, 1023, 511, 0, 1 }, { 3, 1023, 511, 1, 1 },
        { 3, 1023, 511, 2, 1 }, { 3, 1023, 511, 3, 1 } };

INSTANTIATE_TEST_SUITE_P (test_elementwise, TestElementwise,
                          testing::ValuesIn (params));
}
