#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/cast.h"
#include "ops/reduce.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>

namespace vkllama
{
struct TestReduceParams
{
  const int C;
  const int H;
  const int W;
  const int op_type;
  const int dtype;
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

  ASSERT_EQ (command_->begin (), absl::OkStatus ())
      << "failed at begin commands";
  auto input0 = random_tensor<Eigen::half> (gpu_, command_, params.C, params.H,
                                            params.W);

  ASSERT_TRUE (input0) << "failed at create tensor";

  Reduce reduce_op (gpu_, command_, op_type, (Tensor::DType)params.dtype);
  ASSERT_EQ (reduce_op.init (), absl::OkStatus ()) << "failed at init op";

  absl::StatusOr<Tensor> output;
  ASSERT_EQ ((output = reduce_op (input0->first)).status (), absl::OkStatus ())
      << "failed at forwarding reduce op";

  std::vector<Eigen::half> output_buf (output->size ());
  ASSERT_EQ (
      command_->download (*output, output_buf.data (), output_buf.size ()),
      absl::OkStatus ())
      << "failed at download output";

  ASSERT_EQ (command_->end (), absl::OkStatus ()) << "failed at end commands";
  ASSERT_EQ (command_->submit_and_wait (), absl::OkStatus ())
      << "failed at submit commands";

  _Tensor<Eigen::half, 2> vk_output_tensor = _TensorMap<Eigen::half, 2> (
      output_buf.data (), (Eigen::Index)output->channels (),
      (Eigen::Index)output->height ());

  _Tensor<Eigen::half, 3> input0_tensor = _TensorMap<Eigen::half, 3> (
      input0->second.data (), (Eigen::Index)input0->first.channels (),
      (Eigen::Index)input0->first.height (),
      (Eigen::Index)input0->first.width ());

  _Tensor<Eigen::half, 2> output_tensor;
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

  _Tensor<Eigen::half, 2> err (vk_output_tensor.dimensions ());
  err.setConstant (Eigen::half (1e-1));
  _Tensor<int, 0> diff
      = ((vk_output_tensor - output_tensor).abs () > err).cast<int> ().sum ();
  ASSERT_EQ (*diff.data (), 0);
}

std::vector<TestReduceParams> params = {
  { 1, 1023, 511, 0, 1 },
  { 1, 1023, 511, 1, 1 },
  { 1, 1023, 511, 2, 1 },
  { 1, 1023, 511, 3, 1 },
};

INSTANTIATE_TEST_SUITE_P (test_reduce, TestReduce,
                          ::testing::ValuesIn (params));
}
