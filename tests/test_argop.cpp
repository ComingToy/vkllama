#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/argop.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>

namespace
{
struct TestArgOpParams
{
  const uint32_t C;
  const uint32_t H;
  const uint32_t W;
  const int op_type;
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
  const auto params = GetParam ();
  const auto op_type = params.op_type;

  ASSERT_EQ (command_->begin (), VK_SUCCESS);
  auto input0
      = random_tensor<float> (gpu_, command_, params.C, params.H, params.W);

  ASSERT_TRUE (input0);

  VkTensor output;
  ArgMax argmax (gpu_, command_);
  ArgMin argmin (gpu_, command_);
  if (op_type == 0)
    {
      ASSERT_EQ (argmax.init (), VK_SUCCESS);
      ASSERT_EQ (argmax (input0->first, output), VK_SUCCESS);
    }
  else if (op_type == 1)
    {
      ASSERT_EQ (argmin.init (), VK_SUCCESS);
      ASSERT_EQ (argmin (input0->first, output), VK_SUCCESS);
    }

  std::vector<uint32_t> output_buf (output.size ());
  ASSERT_EQ (
      command_->download (output, output_buf.data (), output_buf.size ()),
      VK_SUCCESS);
  ASSERT_EQ (command_->end (), VK_SUCCESS) << "failed at end commands";
  ASSERT_EQ (command_->submit_and_wait (), VK_SUCCESS)
      << "failed at submit commands";

  _Tensor<uint32_t, 2> vk_output_tensor = _TensorMap<uint32_t, 2> (
      output_buf.data (), (Eigen::Index)output.channels (),
      (Eigen::Index)output.height ());

  Tensor<3> input0_tensor = TensorMap<3> (
      input0->second.data (), (Eigen::Index)input0->first.channels (),
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

  // std::cerr << "eigen output: " << eigen_output_tensor << std::endl
  //           << "vk output: " << vk_output_tensor << std::endl;
  _Tensor<uint32_t, 0> diff = (eigen_output_tensor - vk_output_tensor).sum ();
  ASSERT_EQ (*diff.data (), 0);
};

std::vector<TestArgOpParams> params = {
  { 1, 1023, 511, 0 },
  { 1, 1023, 511, 1 },
  { 3, 1023, 511, 0 },
  { 3, 1023, 511, 1 },
};

INSTANTIATE_TEST_SUITE_P (test_argop, TestArgOp, ::testing::ValuesIn (params));
}
