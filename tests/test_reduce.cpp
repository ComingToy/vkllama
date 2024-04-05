#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/reduce.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>

namespace
{
struct TestReduceParams
{
  const int C;
  const int H;
  const int W;
  const int op_type;
};

class TestReduce : public ::testing::TestWithParam<TestReduceParams>
{
public:
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

  GPUDevice *gpu_;
  Command *command_;
};

TEST_P (TestReduce, test_reduce)
{
  auto params = GetParam ();
  const auto op_type = params.op_type;

  ASSERT_EQ (command_->begin (), VK_SUCCESS) << "failed at begin commands";
  auto input0 = random_tensor (gpu_, command_, params.C, params.H, params.W);

  ASSERT_TRUE (input0) << "failed at create tensor";

  Reduce reduce_op (gpu_, command_, op_type);
  ASSERT_EQ (reduce_op.init (), VK_SUCCESS) << "failed at init op";

  VkTensor output;
  ASSERT_EQ (reduce_op (input0->first, output), VK_SUCCESS)
      << "failed at forwarding reduce op";

  std::vector<float> output_buf (output.size ());
  ASSERT_EQ (
      command_->download (output, output_buf.data (), output_buf.size ()),
      VK_SUCCESS)
      << "failed at download output";

  ASSERT_EQ (command_->end (), VK_SUCCESS) << "failed at end commands";
  ASSERT_EQ (command_->submit_and_wait (), VK_SUCCESS)
      << "failed at submit commands";

  Tensor<2> vk_output_tensor = TensorMap<2> (
      output_buf.data (), output.channels (), output.height ());

  Tensor<3> input0_tensor
      = TensorMap<3> (input0->second.data (), input0->first.channels (),
                      input0->first.height (), input0->first.width ());

  Tensor<2> output_tensor;
  Eigen::array<Eigen::Index, 1> dims = { 2 };
  if (op_type == 0)
    {
      output_tensor = input0_tensor.sum (dims);
    }
  else if (op_type == 1)
    {
      output_tensor = input0_tensor.maximum (dims);
    }
  else if (op_type == 2)
    {
      output_tensor = input0_tensor.minimum (dims);
    }
  else if (op_type == 3)
    {
      output_tensor = input0_tensor.mean (dims);
    }

  // std::cerr << "input: " << input0_tensor << std::endl
  //           << "vulkan output: " << vk_output_tensor << std::endl
  //           << "host output: " << output_tensor << std::endl;
  Tensor<0> mse = (output_tensor - vk_output_tensor).pow (2.0).mean ();
  ASSERT_LT (*mse.data (), 1e-4);
}

std::vector<TestReduceParams> params = {
  { 1, 1023, 511, 0 },
  { 1, 1023, 511, 1 },
  { 1, 1023, 511, 2 },
  { 1, 1023, 511, 3 },
};

INSTANTIATE_TEST_SUITE_P (test_reduce, TestReduce,
                          ::testing::ValuesIn (params));
}
