#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/cast.h"
#include "ops/concat.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <memory>
#include <tuple>
#include <vector>

namespace vkllama
{
struct TestConcatParams
{
  const int dtype; // 0: fp32 1: fp16
  const int axis;
  std::vector<std::tuple<int, int, int> > shapes;
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
    ASSERT_EQ (gpu_->init (), absl::OkStatus ());
    ASSERT_EQ (command_->init (), absl::OkStatus ());
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
  ASSERT_EQ (command_->begin (), absl::OkStatus ())
      << "failed at begin commands";
  std::vector<VkTensor> input_tensors, input_tensors_fp16, input_tensors_fp32;
  std::vector<std::unique_ptr<Cast> > cast_input_ops;
  std::vector<std::vector<float> > input_bufs;

  for (auto const &[c, h, w] : params.shapes)
    {
      auto input = random_tensor<float> (gpu_, command_, c, h, w);
      ASSERT_TRUE (input) << "failed at create tensor";
      input_tensors.push_back (input->first);

      if (params.dtype)
        {
          cast_input_ops.emplace_back (
              new Cast (gpu_, command_, VkTensor::FP32, VkTensor::FP16));
          ASSERT_EQ (cast_input_ops.back ()->init (), absl::OkStatus ());
          VkTensor input_fp16;
          ASSERT_EQ (
              cast_input_ops.back ()->operator() (input->first, input_fp16),
              absl::OkStatus ());
          input_tensors_fp16.push_back (input_fp16);

          VkTensor input_fp32;
          cast_input_ops.emplace_back (
              new Cast (gpu_, command_, VkTensor::FP16, VkTensor::FP32));
          ASSERT_EQ (cast_input_ops.back ()->init (), absl::OkStatus ());
          ASSERT_EQ (
              cast_input_ops.back ()->operator() (input_fp16, input_fp32),
              absl::OkStatus ());
          input_tensors_fp32.push_back (input_fp32);

          std::vector<float> input_buf_fp32 (input_fp32.size ());
          ASSERT_EQ (command_->download (input_fp32, input_buf_fp32.data (),
                                         input_buf_fp32.size ()),
                     absl::OkStatus ());
          input_bufs.push_back (std::move (input_buf_fp32));
        }
      else
        {
          input_bufs.push_back (std::move (input->second));
        }
    }

  Concat concat_op (gpu_, command_, (int)input_tensors.size (), params.axis,
                    (VkTensor::DType)params.dtype);
  ASSERT_EQ (concat_op.init (), absl::OkStatus ()) << "failed at init op";

  VkTensor output, output_fp16;
  Cast cast_output_op (gpu_, command_, VkTensor::FP16, VkTensor::FP32);
  ASSERT_EQ (cast_output_op.init (), absl::OkStatus ());
  if (params.dtype)
    {
      ASSERT_EQ (concat_op (input_tensors_fp16, output_fp16),
                 absl::OkStatus ());
      ASSERT_EQ (cast_output_op (output_fp16, output), absl::OkStatus ());
    }
  else
    {
      ASSERT_EQ (concat_op (input_tensors, output), absl::OkStatus ())
          << "failed at infer concat";
    }
  std::vector<float> output_buf (output.size ());

  ASSERT_EQ (
      command_->download (output, output_buf.data (), output_buf.size ()),
      absl::OkStatus ())
      << "failed at download output";
  ASSERT_EQ (command_->end (), absl::OkStatus ()) << "failed at end commands";
  ASSERT_EQ (command_->submit_and_wait (), absl::OkStatus ())
      << "failed at submit commands";

  std::vector<Tensor<3> > input_eigen_tensors (input_tensors.size ());
  for (int i = 0; i < input_eigen_tensors.size (); ++i)
    {
      auto const &t = input_tensors[i];
      input_eigen_tensors[i]
          = TensorMap<3> (input_bufs[i].data (), (Eigen::Index)t.channels (),
                          (Eigen::Index)t.height (), (Eigen::Index)t.width ());
    }

  Tensor<3> tmp = input_eigen_tensors[0];
  Tensor<3> eigen_output_tensor;
  for (int i = 1; i < input_eigen_tensors.size (); ++i)
    {
      eigen_output_tensor
          = tmp.concatenate (input_eigen_tensors[i], params.axis);
      tmp = eigen_output_tensor;
    }

  // Tensor<3> eigen_output_tensor = concat_expr;

  Tensor<3> vk_output_tensor = TensorMap<3> (
      output_buf.data (), (Eigen::Index)output.channels (),
      (Eigen::Index)output.height (), (Eigen::Index)output.width ());

  Tensor<3> err (vk_output_tensor.dimensions ());
  err.setConstant (params.dtype ? 1e-2 : 1e-3);
  _Tensor<int, 0> diff
      = ((vk_output_tensor - eigen_output_tensor).abs () > err)
            .cast<int> ()
            .sum ();
  ASSERT_EQ (*diff.data (), 0);
}

std::vector<TestConcatParams> params = {
  { 0, 0, { { 1, 63, 31 }, { 2, 63, 31 } } },
  { 0, 1, { { 3, 63, 31 }, { 3, 31, 31 } } },
  { 0, 2, { { 3, 63, 31 }, { 3, 63, 63 } } },

  { 1, 0, { { 1, 63, 31 }, { 2, 63, 31 }, { 5, 63, 31 } } },
  { 1, 1, { { 3, 63, 31 }, { 3, 31, 31 }, { 3, 22, 31 } } },
  { 1, 2, { { 3, 63, 31 }, { 3, 63, 63 }, { 3, 63, 22 } } },
};

INSTANTIATE_TEST_SUITE_P (test_concat, TestConcat,
                          ::testing::ValuesIn (params));
}
