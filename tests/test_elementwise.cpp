#include "Eigen/Eigen"
#include "core/command.h"
#include "core/float.h"
#include "core/gpu_device.h"
#include "ops/cast.h"
#include "ops/elementwise.h"
#include "tests/test_common.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <vector>

namespace vkllama
{
struct TestElementwiseParams
{
  const int C;
  const int H;
  const int W;
  const int op_type;
  const bool constant_b;
  const int dtype; // 0: fp32, 1: fp16
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
  ASSERT_EQ (command_->begin (), absl::OkStatus ())
      << "failed at begining commands";
  auto input0
      = random_tensor<float> (gpu_, command_, params.C, params.H, params.W);
  auto input1
      = random_tensor<float> (gpu_, command_, params.C, params.H, params.W);
  float alpha = random_number (-2.0f, 2.0f);

  VkTensor input0_fp16;
  VkTensor input1_fp16;
  Cast cast_input_op0 (gpu_, command_, VkTensor::FP32, VkTensor::FP16);
  Cast cast_input_op1 (gpu_, command_, VkTensor::FP32, VkTensor::FP16);
  Cast cast_output_op (gpu_, command_, VkTensor::FP16, VkTensor::FP32);
  ASSERT_EQ (cast_input_op0.init (), absl::OkStatus ());
  ASSERT_EQ (cast_input_op1.init (), absl::OkStatus ());
  ASSERT_EQ (cast_output_op.init (), absl::OkStatus ());

  if (params.dtype == 1)
    {
      ASSERT_EQ (cast_input_op0 (input0->first, input0_fp16),
                 absl::OkStatus ());
      ASSERT_EQ (cast_input_op1 (input1->first, input1_fp16),
                 absl::OkStatus ());
    }

  ElementWise elementwise_op (gpu_, command_, params.op_type,
                              params.dtype == 0 ? VkTensor::FP32
                                                : VkTensor::FP16);
  ASSERT_EQ (elementwise_op.init (), absl::OkStatus ())
      << "failed at init elementwise op";

  VkTensor out;
  VkTensor out_fp16;
  if (params.dtype == 0)
    {
      if (params.constant_b)
        {
          ASSERT_EQ (elementwise_op (input0->first, alpha, out),
                     absl::OkStatus ());
        }
      else
        {
          ASSERT_EQ (elementwise_op (input0->first, input1->first, out),
                     absl::OkStatus ());
        }
    }
  else
    {
      if (params.constant_b)
        {
          ASSERT_EQ (elementwise_op (input0_fp16, alpha, out_fp16),
                     absl::OkStatus ());
        }
      else
        {
          ASSERT_EQ (elementwise_op (input0_fp16, input1_fp16, out_fp16),
                     absl::OkStatus ());
        }

      ASSERT_EQ (cast_output_op (out_fp16, out), absl::OkStatus ());
    }

  std::vector<float> output_buf (out.size ());
  ASSERT_EQ (command_->download (out, output_buf.data (), output_buf.size ()),
             absl::OkStatus ())
      << "failed at download output tensor";

  ASSERT_EQ (command_->end (), absl::OkStatus ())
      << "failed at edndding commands";
  ASSERT_EQ (command_->submit_and_wait (), absl::OkStatus ())
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

  Tensor<3> err (vk_output_tensor.dimensions ());

  auto delta = params.dtype ? 1e-2 : 1e-3;
#if 0
  for (size_t i = 0; i < vk_output_tensor.size (); ++i)
    {
      auto e = fabs (output_tensor (i) - vk_output_tensor (i));
      if (e < delta)
        {
          continue;
        }

      fprintf (
          stderr,
          "index %zu inpu0 = %f, input1 = %f, lhs = %f, rhs = %f, err = %f\n",
          i, input0->second[i], params.constant_b ? alpha : input1->second[i],
          output_tensor (i), vk_output_tensor (i), e);
    }
#endif

  err.setConstant (delta);
  _Tensor<int, 0> diff
      = ((vk_output_tensor - output_tensor).abs () > err).cast<int> ().sum ();

  ASSERT_EQ (*diff.data (), 0);
}

#if 1
std::vector<TestElementwiseParams> params
    = { { 3, 1023, 512, 0, 0, 0 }, { 3, 1023, 511, 1, 0, 0 },
        { 3, 1023, 511, 2, 0, 0 }, { 3, 1023, 511, 3, 0, 0 },
        { 3, 1023, 511, 0, 1, 0 }, { 3, 1023, 511, 1, 1, 0 },
        { 3, 1023, 511, 2, 1, 0 }, { 3, 1023, 511, 3, 1, 0 },

        { 3, 1023, 512, 0, 0, 1 }, { 3, 1023, 511, 1, 0, 1 },
        { 3, 1023, 511, 2, 0, 1 }, { 3, 1023, 511, 3, 0, 1 },
        { 3, 1023, 511, 0, 1, 1 }, { 3, 1023, 511, 1, 1, 1 },
        { 3, 1023, 511, 2, 1, 1 }, { 3, 1023, 511, 3, 1, 1 } };
#else

std::vector<TestElementwiseParams> params
    = { { 3, 1023, 511, 0, 1, 1 }, { 3, 1023, 511, 1, 1, 1 } };
#endif

INSTANTIATE_TEST_SUITE_P (test_elementwise, TestElementwise,
                          testing::ValuesIn (params));
}
