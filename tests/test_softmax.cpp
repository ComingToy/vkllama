#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/softmax.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>

namespace
{
struct TestSoftmaxParams
{

  const int C;
  const int H;
  const int W;
};

class TestSoftmax : public ::testing::TestWithParam<TestSoftmaxParams>
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

TEST_P (TestSoftmax, test_softmax)
{
  auto params = GetParam ();
  ASSERT_EQ (command_->begin (), VK_SUCCESS) << "failed at begin commands";

  auto input0 = random_tensor (gpu_, command_, params.C, params.H, params.W);
  ASSERT_TRUE (input0) << "failed at create tensor";

  Softmax softmax_op (gpu_, command_);
  ASSERT_EQ (softmax_op.init (), VK_SUCCESS) << "failed at init op";

  VkTensor output;
  ASSERT_EQ (softmax_op (input0->first, output), VK_SUCCESS)
      << "failed at infer softmax";

  std::vector<float> output_buf (output.size ());
  ASSERT_EQ (
      command_->download (output, output_buf.data (), output_buf.size ()),
      VK_SUCCESS)
      << "failed at download output";

  ASSERT_EQ (command_->end (), VK_SUCCESS) << "failed at end commands";
  ASSERT_EQ (command_->submit_and_wait (), VK_SUCCESS)
      << "failed at submit commands";

  Tensor<3> vk_output_tensor
      = TensorMap<3> (output_buf.data (), output.channels (), output.height (),
                      output.width ());

  Tensor<3> input_tensor
      = TensorMap<3> (input0->second.data (), input0->first.channels (),
                      input0->first.height (), input0->first.width ());
  Tensor<3> output_tensor (input_tensor.dimensions ());
  Eigen::array<Eigen::Index, 1> dims = { 2 };

  Tensor<3> exps;
  float m;
  {
    Eigen::array<Eigen::Index, 3> bias_dims
        = { input_tensor.dimension (0), input_tensor.dimension (1), 1 };
    Eigen::array<Eigen::Index, 3> broadcasts
        = { 1, 1, input_tensor.dimension (2) };
    auto debias = input_tensor
                  - input_tensor.maximum (dims).reshape (bias_dims).broadcast (
                      broadcasts);
    exps = debias.exp ();
    Tensor<0> sum = exps.sum ();
    m = *sum.data ();
  }

  output_tensor = exps / m;
  Tensor<0> mse = (vk_output_tensor - output_tensor).pow (2.0).mean ();
  ASSERT_LT (*mse.data (), 1e-4);
  // std::cerr << "input tensor: " << input_tensor << std::endl
  //           << "output_tensor: " << output_tensor << std::endl
  //           << "vk_output_tensor: " << vk_output_tensor << std::endl;
  // std::cerr << "mean of output_tensor: " << output_tensor.mean () <<
  // std::endl; std::cerr << "mean of vk_output_tensor: " <<
  // vk_output_tensor.mean ()
  //           << std::endl;
}

std::vector<TestSoftmaxParams> params = {
  { 1, 1024, 512 }, { 3, 1024, 512 }, { 1, 1023, 511 }, { 3, 1023, 511 }
};

INSTANTIATE_TEST_SUITE_P (test_softmax, TestSoftmax,
                          ::testing::ValuesIn (params));
};

