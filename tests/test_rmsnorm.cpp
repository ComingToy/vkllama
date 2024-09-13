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
  ASSERT_EQ (command_->begin (), absl::OkStatus ())
      << "failed at begin commands";

  auto params = GetParam ();
  auto input0 = random_tensor<Eigen::half> (gpu_, command_, params.C, params.H,
                                            params.W);
  auto input1 = random_tensor<float> (gpu_, command_, 1, 1, params.W, -1.0f,
                                      1.0f, Tensor::FP32);

  ASSERT_TRUE (input0);
  ASSERT_TRUE (input1);

  RMSNorm norm_op (gpu_, command_, input1->first, 1e-3f,
                   (Tensor::DType)params.dtype);
  ASSERT_EQ (norm_op.init (), absl::OkStatus ());

  absl::StatusOr<Tensor> output;
  ASSERT_EQ ((output = norm_op (input0->first)).status (), absl::OkStatus ());

  std::vector<Eigen::half> output_buf (output->size ());

  ASSERT_EQ (command_->download (*output,
                                 (__vkllama_fp16_t *)output_buf.data (),
                                 output_buf.size ()),
             absl::OkStatus ());

  ASSERT_EQ (command_->end (), absl::OkStatus ()) << "failed at end commands";
  ASSERT_EQ (command_->submit_and_wait (), absl::OkStatus ())
      << "failed at submit commands";

  _Tensor<Eigen::half, 3> vk_output_tensor = _TensorMap<Eigen::half, 3> (
      output_buf.data (), (Eigen::Index)output->channels (),
      (Eigen::Index)output->height (), (Eigen::Index)output->width ());

  _Tensor<Eigen::half, 3> input_tensor0 = _TensorMap<Eigen::half, 3> (
      input0->second.data (), (Eigen::Index)input0->first.channels (),
      (Eigen::Index)input0->first.height (),
      (Eigen::Index)input0->first.width ());

  _Tensor<float, 3> input_tensor1 = _TensorMap<float, 3> (
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

  _Tensor<Eigen::half, 3> eigen_output_tensor
      = ((input_tensor0.pow (Eigen::half (2.0f)).mean (mean_dims)
          + Eigen::half (1e-3f))
             .rsqrt ()
             .reshape (dims)
             .broadcast (broadcasts)
             .cast<float> ()
         * input_tensor1.broadcast (weight_broadcasts)
         * input_tensor0.cast<float> ())
            .cast<Eigen::half> ();
  // std::cerr << "input tensor: " << input_tensor0 << std::endl
  //           << "vk output tensor: " << vk_output_tensor << std::endl
  //           << "eigen output tensor: " << eigen_output_tensor << std::endl;
  _Tensor<Eigen::half, 3> err (vk_output_tensor.dimensions ());
  err.setConstant (Eigen::half (1e-2));

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
