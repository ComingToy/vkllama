#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/cast.h"
#include "ops/mat_mul.h"
#include "test_common.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "gtest/gtest.h"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <math.h>
#include <vector>

namespace vkllama
{
struct TestMatMulParams
{
  const int C;
  const int M;
  const int N;
  const int K;
  const int broadcast_type;
  const int transpose_b;
  const int dtype; // 0: fp32 1: fp16
};

class TestMatmul : public ::testing::TestWithParam<TestMatMulParams>
{
public:
  void
  SetUp () override
  {
    gpu_ = new GPUDevice ();
    gpu_->init ();

    command_ = new Command (gpu_);
    command_->init ();
  };

  void
  TearDown () override
  {
    delete command_;
    delete gpu_;
  }

  GPUDevice *gpu_;
  Command *command_;
};

TEST_P (TestMatmul, test_matmul_broadcast)
{
  auto params = GetParam ();
  const auto broadcast_type = params.broadcast_type;
  int in0_channel = 0, in1_channel = 0;

  if (broadcast_type == 0)
    {
      in0_channel = params.C;
      in1_channel = params.C;
    }
  else if (broadcast_type == 1)
    {
      in0_channel = params.C;
      in1_channel = 1;
    }
  else if (broadcast_type == 2)
    {
      in0_channel = 1;
      in1_channel = params.C;
    }
  else
    {
      ASSERT_TRUE (false) << "unsupported broadcast_type " << broadcast_type;
    }

  int in0_h = params.M, in0_w = params.K, in1_h = params.K, in1_w = params.N;
  if (params.transpose_b > 0)
    {
      in1_h = params.N;
      in1_w = params.K;
    }

  ASSERT_EQ (command_->begin (), VK_SUCCESS) << "failed at begin command";
  auto input0
      = random_tensor<float> (gpu_, command_, in0_channel, in0_h, in0_w);
  auto input1
      = random_tensor<float> (gpu_, command_, in1_channel, in1_h, in1_w);

  ASSERT_TRUE (input0 && input1) << "failed at create tensors";

  VkTensor input0_fp16, input1_fp16, input0_fp32, input1_fp32;
  std::vector<float> input0_buf, input1_buf;
  Cast cast_input0_op (gpu_, command_, VkTensor::FP32, VkTensor::FP16);
  Cast cast_input1_op (gpu_, command_, VkTensor::FP32, VkTensor::FP16);

  Cast cast_input0_op_fp32 (gpu_, command_, VkTensor::FP16, VkTensor::FP32);
  Cast cast_input1_op_fp32 (gpu_, command_, VkTensor::FP16, VkTensor::FP32);

  if (params.dtype == 1)
    {
      ASSERT_EQ (cast_input0_op.init (), VK_SUCCESS);
      ASSERT_EQ (cast_input1_op.init (), VK_SUCCESS);
      ASSERT_EQ (cast_input0_op_fp32.init (), VK_SUCCESS);
      ASSERT_EQ (cast_input1_op_fp32.init (), VK_SUCCESS);

      ASSERT_EQ (cast_input0_op (input0->first, input0_fp16), VK_SUCCESS);
      ASSERT_EQ (cast_input1_op (input1->first, input1_fp16), VK_SUCCESS);
      ASSERT_EQ (cast_input0_op_fp32 (input0_fp16, input0_fp32), VK_SUCCESS);
      ASSERT_EQ (cast_input1_op_fp32 (input1_fp16, input1_fp32), VK_SUCCESS);
      input0_buf.resize (input0_fp16.size ());
      input1_buf.resize (input1_fp16.size ());
      ASSERT_EQ (command_->download (input0_fp32, input0_buf.data (),
                                     input0_buf.size ()),
                 VK_SUCCESS);

      ASSERT_EQ (command_->download (input1_fp32, input1_buf.data (),
                                     input1_buf.size ()),
                 VK_SUCCESS);
    }
  else
    {
      input0_fp32 = input0->first;
      input1_fp32 = input1->first;
      input0_buf.swap (input0->second);
      input1_buf.swap (input1->second);
    }

  MatMul matmul_op (gpu_, command_, 1.0, .0, 0, params.broadcast_type,
                    params.transpose_b, (VkTensor::DType)params.dtype);

  ASSERT_TRUE (matmul_op.init () == VK_SUCCESS) << "failed at init matmul op";

  VkTensor output;
  ASSERT_TRUE (matmul_op (params.dtype ? input0_fp16 : input0_fp32,
                          params.dtype ? input1_fp16 : input1_fp32, output)
               == VK_SUCCESS)
      << "failed at forwarding matmul op";

  VkTensor output_fp32;
  Cast cast_output_op (gpu_, command_, VkTensor::FP16, VkTensor::FP32);

  if (params.dtype)
    {
      ASSERT_EQ (cast_output_op.init (), VK_SUCCESS);
      ASSERT_EQ (cast_output_op (output, output_fp32), VK_SUCCESS);
    }
  else
    {
      output_fp32 = output;
    }

  std::vector<float> buf (output_fp32.size ());
  ASSERT_EQ (command_->download (output_fp32, buf.data (), buf.size ()),
             VK_SUCCESS)
      << "failed at downloading output";
  ASSERT_EQ (command_->end (), VK_SUCCESS) << "failed at endding commands";
  ASSERT_EQ (command_->submit_and_wait (), VK_SUCCESS)
      << "failed at submiting commands";

  // eigen matmul
  using TensorMap
      = Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor> >;
  Tensor<3> in0_tensor
      = TensorMap (input0_buf.data (), (Eigen::Index)input0->first.channels (),
                   (Eigen::Index)input0->first.height (),
                   (Eigen::Index)input0->first.width ());
  Tensor<3> in1_tmp
      = TensorMap (input1_buf.data (), (Eigen::Index)input1->first.channels (),
                   (Eigen::Index)input1->first.height (),
                   (Eigen::Index)input1->first.width ());
  Tensor<3> in1_tensor;

  if (params.transpose_b)
    {
      Eigen::array<Eigen::Index, 3> trans = { 0, 2, 1 };
      in1_tensor = in1_tmp.shuffle (trans);
    }
  else
    {
      in1_tensor = in1_tmp;
    }

  Eigen::Tensor<float, 3, Eigen::RowMajor> eigen_output (
      (Eigen::Index)output.channels (), (Eigen::Index)output.height (),
      (Eigen::Index)output.width ());
  Eigen::array<Eigen::IndexPair<int>, 1> dims
      = { Eigen::IndexPair<int> (1, 0) };
  if (broadcast_type == 0)
    {
      for (int i = 0; i < output.channels (); ++i)
        {
          eigen_output.chip<0> (i)
              = in0_tensor.chip<0> (i).contract (in1_tensor.chip<0> (i), dims);
        }
    }
  else if (broadcast_type == 1)
    {
      for (int i = 0; i < output.channels (); ++i)
        {
          eigen_output.chip<0> (i)
              = in0_tensor.chip<0> (i).contract (in1_tensor.chip<0> (0), dims);
        }
    }
  else if (broadcast_type == 2)
    {
      for (int i = 0; i < output.channels (); ++i)
        {
          eigen_output.chip<0> (i)
              = in0_tensor.chip<0> (0).contract (in1_tensor.chip<0> (i), dims);
        }
    }

  auto output_mapped = TensorMap (
      buf.data (), (Eigen::Index)output.channels (),
      (Eigen::Index)output.height (), (Eigen::Index)output.width ());

  Tensor<3> err (eigen_output.dimensions ());
  err.setConstant (1e-2);
  _Tensor<int, 0> diff
      = ((eigen_output - output_mapped).abs () > err).cast<int> ().sum ();
  ASSERT_EQ (*diff.data (), 0);
}

std::vector<TestMatMulParams> params = {
  { 1, 1024, 1023, 225, 0, 1, 0 }, { 1, 1027, 619, 32, 0, 1, 0 },
  { 5, 1024, 512, 256, 0, 1, 0 },  { 16, 255, 321, 513, 0, 1, 0 },
  { 1, 1024, 1023, 225, 1, 1, 0 }, { 1, 1027, 619, 32, 1, 1, 0 },
  { 5, 1024, 512, 256, 1, 1, 0 },  { 16, 255, 321, 513, 1, 1, 0 },
  { 1, 1024, 1023, 225, 2, 1, 0 }, { 1, 1027, 619, 32, 2, 1, 0 },
  { 5, 1024, 512, 256, 2, 1, 0 },  { 16, 255, 321, 513, 2, 1, 0 },

  { 1, 1024, 1023, 225, 0, 0, 0 }, { 1, 1027, 619, 32, 0, 0, 0 },
  { 5, 1024, 512, 256, 0, 0, 0 },  { 16, 255, 321, 513, 0, 0, 0 },
  { 1, 1024, 1023, 225, 1, 0, 0 }, { 1, 1027, 619, 32, 1, 0, 0 },
  { 5, 1024, 512, 256, 1, 0, 0 },  { 16, 255, 321, 513, 1, 0, 0 },
  { 1, 1024, 1023, 225, 2, 0, 0 }, { 1, 1027, 619, 32, 2, 0, 0 },
  { 5, 1024, 512, 256, 2, 0, 0 },  { 16, 255, 321, 513, 2, 0, 0 },

  { 1, 1024, 1023, 225, 0, 1, 1 }, { 1, 1027, 619, 32, 0, 1, 1 },
  { 5, 1024, 512, 256, 0, 1, 1 },  { 16, 255, 321, 513, 0, 1, 1 },
  { 1, 1024, 1023, 225, 1, 1, 1 }, { 1, 1027, 619, 32, 1, 1, 1 },
  { 5, 1024, 512, 256, 1, 1, 1 },  { 16, 255, 321, 513, 1, 1, 1 },
  { 1, 1024, 1023, 225, 2, 1, 1 }, { 1, 1027, 619, 32, 2, 1, 1 },
  { 5, 1024, 512, 256, 2, 1, 1 },  { 16, 255, 321, 513, 2, 1, 1 },

  { 1, 1024, 1023, 225, 0, 0, 1 }, { 1, 1027, 619, 32, 0, 0, 1 },
  { 5, 1024, 512, 256, 0, 0, 1 },  { 16, 255, 321, 513, 0, 0, 1 },
  { 1, 1024, 1023, 225, 1, 0, 1 }, { 1, 1027, 619, 32, 1, 0, 1 },
  { 5, 1024, 512, 256, 1, 0, 1 },  { 16, 255, 321, 513, 1, 0, 1 },
  { 1, 1024, 1023, 225, 2, 0, 1 }, { 1, 1027, 619, 32, 2, 0, 1 },
  { 5, 1024, 512, 256, 2, 0, 1 },  { 16, 255, 321, 513, 2, 0, 1 },
};

INSTANTIATE_TEST_SUITE_P (TestMatmulBroadcast, TestMatmul,
                          testing::ValuesIn (params));
} // namespace

