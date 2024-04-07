#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/concat.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>

namespace
{
struct TestConcatParams
{
  const int C;
  const int H;
  const std::vector<int> W;
};

class TestConcat : public ::testing::TestWithParam<TestConcatParams>
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

TEST_P (TestConcat, test_concat)
{
  auto params = GetParam ();
  ASSERT_EQ (command_->begin (), VK_SUCCESS) << "failed at begin commands";
  std::vector<VkTensor> input_tensors;
  std::vector<std::vector<float> > input_bufs;

  for (auto const w : params.W)
    {
      auto input = random_tensor (gpu_, command_, params.C, params.H, w);
      ASSERT_TRUE (input) << "failed at create tensor";
      input_tensors.push_back (input->first);
      input_bufs.push_back (std::move (input->second));
    }

  Concat concat_op (gpu_, command_, (int)input_tensors.size ());
  ASSERT_EQ (concat_op.init (), VK_SUCCESS) << "failed at init op";

  VkTensor output;
  ASSERT_EQ (concat_op (input_tensors, output), VK_SUCCESS)
      << "failed at infer concat";
  std::vector<float> output_buf (output.size ());

  ASSERT_EQ (
      command_->download (output, output_buf.data (), output_buf.size ()),
      VK_SUCCESS)
      << "failed at download output";
  ASSERT_EQ (command_->end (), VK_SUCCESS) << "failed at end commands";
  ASSERT_EQ (command_->submit_and_wait (), VK_SUCCESS)
      << "failed at submit commands";

  std::vector<Tensor<3> > input_eigen_tensors (input_tensors.size ());
  for (int i = 0; i < input_eigen_tensors.size (); ++i)
    {
      auto const &t = input_tensors[i];
      input_eigen_tensors[i] = TensorMap<3> (
          input_bufs[i].data (), t.channels (), t.height (), t.width ());
    }

  Tensor<3> tmp = input_eigen_tensors[0];
  Tensor<3> eigen_output_tensor;
  for (int i = 1; i < input_eigen_tensors.size (); ++i)
    {
      eigen_output_tensor = tmp.concatenate (input_eigen_tensors[i], 2);
      tmp = eigen_output_tensor;
    }

  // Tensor<3> eigen_output_tensor = concat_expr;

  Tensor<3> vk_output_tensor
      = TensorMap<3> (output_buf.data (), output.channels (), output.height (),
                      output.width ());
  Tensor<0> mse = (vk_output_tensor - eigen_output_tensor).pow (2).mean ();
#if 0
  for (auto i = 0; i < input_eigen_tensors.size (); ++i)
    {
      std::cerr << "input tensor" << i << ": " << input_eigen_tensors[i]
                << std::endl;
    }

  std::cerr << "eigen output tensor: " << eigen_output_tensor << std::endl
            << "vk output tensor: " << vk_output_tensor << std::endl;
#endif
  ASSERT_LT (*mse.data (), 1e-4);
}

std::vector<TestConcatParams> params = {
  { 1, 64, { 32, 32 } },      { 5, 64, { 32, 32 } },
  { 1, 65, { 32, 32 } },      { 5, 65, { 32, 32 } },
  { 5, 65, { 15, 18 } },      { 5, 65, { 15, 32, 125 } },
  { 5, 65, { 32, 15, 128 } }, { 5, 65, { 33, 128, 15 } },
  { 5, 65, { 128, 19, 31 } },
};

INSTANTIATE_TEST_SUITE_P (test_concat, TestConcat,
                          ::testing::ValuesIn (params));
}
