#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/cast.h"
#include "ops/softmax.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>

namespace vkllama
{
struct TestSoftmaxParams
{
  const int C;
  const int H;
  const int W;
  const int dtype;
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
  ASSERT_EQ (command_->begin (), absl::OkStatus ())
      << "failed at begin commands";

  auto input0 = random_tensor<Eigen::half> (gpu_, command_, params.C, params.H,
                                            params.W);
  ASSERT_TRUE (input0) << "failed at create tensor";

  Softmax softmax_op (gpu_, command_, false, 1.0, (Tensor::DType)params.dtype);
  ASSERT_EQ (softmax_op.init (), absl::OkStatus ()) << "failed at init op";

  absl::StatusOr<Tensor> output;
  ASSERT_EQ ((output = softmax_op (input0->first)).status (),
             absl::OkStatus ())
      << "failed at infer softmax";

  std::vector<Eigen::half> output_buf (output->size ());

  ASSERT_EQ (
      command_->download (*output, output_buf.data (), output_buf.size ()),
      absl::OkStatus ())
      << "failed at download output";

  ASSERT_EQ (command_->end (), absl::OkStatus ()) << "failed at end commands";
  ASSERT_EQ (command_->submit_and_wait (), absl::OkStatus ())
      << "failed at submit commands";

  _Tensor<Eigen::half, 3> vk_output_tensor = _TensorMap<Eigen::half, 3> (
      output_buf.data (), (Eigen::Index)output->channels (),
      (Eigen::Index)output->height (), (Eigen::Index)output->width ());

  _Tensor<Eigen::half, 3> input_tensor = _TensorMap<Eigen::half, 3> (
      input0->second.data (), (Eigen::Index)input0->first.channels (),
      (Eigen::Index)input0->first.height (),
      (Eigen::Index)input0->first.width ());
  _Tensor<Eigen::half, 3> output_tensor (input_tensor.dimensions ());
  Eigen::array<Eigen::Index, 1> dims = { 2 };

  _Tensor<Eigen::half, 3> exps;
  _Tensor<Eigen::half, 3> m;
  {
    Eigen::array<Eigen::Index, 3> bias_dims
        = { input_tensor.dimension (0), input_tensor.dimension (1), 1 };
    Eigen::array<Eigen::Index, 3> broadcasts
        = { 1, 1, input_tensor.dimension (2) };
    auto debias = input_tensor
                  - input_tensor.maximum (dims).reshape (bias_dims).broadcast (
                      broadcasts);
    exps = debias.exp ();
    m = exps.sum (dims).reshape (bias_dims).broadcast (broadcasts);
  }

  output_tensor = exps / m;
  // std::cerr << "input tensor: " << input_tensor << std::endl
  //           << "eigen output tensor: " << output_tensor << std::endl
  //           << "vk output tensor: " << vk_output_tensor << std::endl;

  _Tensor<Eigen::half, 3> err (vk_output_tensor.dimensions ());
  err.setConstant (Eigen::half (1e-2));
  _Tensor<int, 0> diff
      = ((vk_output_tensor - output_tensor).abs () > err).cast<int> ().sum ();
  ASSERT_EQ (*diff.data (), 0);
}

std::vector<TestSoftmaxParams> params = {
  { 32, 62, 62, 1 },
  { 1, 1023, 63, 1 },
  { 3, 1023, 51, 1 },
};

INSTANTIATE_TEST_SUITE_P (test_softmax, TestSoftmax,
                          ::testing::ValuesIn (params));
};

