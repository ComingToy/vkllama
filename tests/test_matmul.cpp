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
  const int a_dtype; // 0: fp32 1: fp16 2: int_8 3: q8_0
  const int b_dtype; // 0: fp32 1: fp16 2: int_8 3: q8_0
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

  ASSERT_EQ (command_->begin (), absl::OkStatus ())
      << "failed at begin command";

  auto input0 = random_tensor<Eigen::half> (
      gpu_, command_, in0_channel, in0_h, in0_w, Eigen::half (-1),
      Eigen::half (1), vkllama::Tensor::DType (params.a_dtype));

  auto input1 = random_tensor<Eigen::half> (
      gpu_, command_, in1_channel, in1_h, in1_w, Eigen::half (-1),
      Eigen::half (1), Tensor::DType (params.b_dtype));

  ASSERT_TRUE (input0 && input1) << "failed at create tensors";

  MatMul matmul_op (gpu_, command_, 1.0, .0, 0, params.broadcast_type,
                    params.transpose_b, (Tensor::DType)params.a_dtype,
                    (Tensor::DType)params.b_dtype);

  absl::Status ret;
  ASSERT_TRUE ((ret = matmul_op.init ()) == absl::OkStatus ())
      << "failed at init matmul op: " << ret;

  absl::StatusOr<Tensor> output = matmul_op (input0->first, input1->first);

  ASSERT_TRUE (output.ok ()) << "failed at forwarding matmul op";

  std::vector<Eigen::half> buf (output->size ());
  ASSERT_EQ (command_->download (*output, (__vkllama_fp16_t *)buf.data (),
                                 buf.size ()),
             absl::OkStatus ())
      << "failed at downloading output";
  ASSERT_EQ (command_->end (), absl::OkStatus ())
      << "failed at endding commands";
  ASSERT_EQ (command_->submit_and_wait (), absl::OkStatus ())
      << "failed at submiting commands";

  // eigen matmul
  using TensorMap
      = Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor> >;

  _Tensor<Eigen::half, 3> in0_tensor = TensorMap (
      input0->second.data (), (Eigen::Index)input0->first.channels (),
      (Eigen::Index)input0->first.height (),
      (Eigen::Index)input0->first.width ());
  _Tensor<Eigen::half, 3> in1_tmp = TensorMap (
      input1->second.data (), (Eigen::Index)input1->first.channels (),
      (Eigen::Index)input1->first.height (),
      (Eigen::Index)input1->first.width ());
  _Tensor<Eigen::half, 3> in1_tensor;

  if (params.transpose_b)
    {
      Eigen::array<Eigen::Index, 3> trans = { 0, 2, 1 };
      in1_tensor = in1_tmp.shuffle (trans);
    }
  else
    {
      in1_tensor = in1_tmp;
    }

  Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor> eigen_output (
      (Eigen::Index)output->channels (), (Eigen::Index)output->height (),
      (Eigen::Index)output->width ());

  Eigen::array<Eigen::IndexPair<int>, 1> dims
      = { Eigen::IndexPair<int> (1, 0) };
  if (broadcast_type == 0)
    {
      for (int i = 0; i < output->channels (); ++i)
        {
          eigen_output.chip<0> (i)
              = in0_tensor.chip<0> (i).contract (in1_tensor.chip<0> (i), dims);
        }
    }
  else if (broadcast_type == 1)
    {
      for (int i = 0; i < output->channels (); ++i)
        {
          eigen_output.chip<0> (i)
              = in0_tensor.chip<0> (i).contract (in1_tensor.chip<0> (0), dims);
        }
    }
  else if (broadcast_type == 2)
    {
      for (int i = 0; i < output->channels (); ++i)
        {
          eigen_output.chip<0> (i)
              = in0_tensor.chip<0> (0).contract (in1_tensor.chip<0> (i), dims);
        }
    }

  auto output_mapped = TensorMap (
      buf.data (), (Eigen::Index)output->channels (),
      (Eigen::Index)output->height (), (Eigen::Index)output->width ());

  _Tensor<Eigen::half, 3> err (eigen_output.dimensions ());
  err.setConstant (Eigen::half (1e-1));
  _Tensor<int, 0> diff
      = ((eigen_output - output_mapped).abs () > err).cast<int> ().sum ();

  // std::cerr << "eigen output: " << eigen_output << std::endl
  //           << "vulkan output: " << output_mapped << std::endl;
  ASSERT_EQ (*diff.data (), 0);
}

std::vector<TestMatMulParams> params = {
  { 1, 10, 122, 111, 0, 1, 1, 4 },    { 1, 512, 128, 64, 0, 1, 1, 4 },
// { 1, 1023, 235, 95, 0, 1, 1, 4 },
#if 1
  { 1, 1024, 1023, 225, 0, 1, 1, 1 }, { 1, 1027, 619, 32, 0, 1, 1, 1 },
  { 5, 1024, 512, 256, 0, 1, 1, 1 },  { 16, 255, 321, 513, 0, 1, 1, 1 },
  { 1, 1024, 1023, 225, 1, 1, 1, 1 }, { 1, 1027, 619, 32, 1, 1, 1, 1 },
  { 5, 1024, 512, 256, 1, 1, 1, 1 },  { 16, 255, 321, 513, 1, 1, 1, 1 },
  { 1, 1024, 1023, 225, 2, 1, 1, 1 }, { 1, 1027, 619, 32, 2, 1, 1, 1 },
  { 5, 1024, 512, 256, 2, 1, 1, 1 },  { 16, 255, 321, 513, 2, 1, 1, 1 },

  { 1, 1024, 1023, 225, 0, 0, 1, 1 }, { 1, 1027, 619, 32, 0, 0, 1, 1 },
  { 5, 1024, 512, 256, 0, 0, 1, 1 },  { 16, 255, 321, 513, 0, 0, 1, 1 },
  { 1, 1024, 1023, 225, 1, 0, 1, 1 }, { 1, 1027, 619, 32, 1, 0, 1, 1 },
  { 5, 1024, 512, 256, 1, 0, 1, 1 },  { 16, 255, 321, 513, 1, 0, 1, 1 },
  { 1, 1024, 1023, 225, 2, 0, 1, 1 }, { 1, 1027, 619, 32, 2, 0, 1, 1 },
  { 5, 1024, 512, 256, 2, 0, 1, 1 },  { 16, 255, 321, 513, 2, 0, 1, 1 },
#endif
};

INSTANTIATE_TEST_SUITE_P (TestMatmulBroadcast, TestMatmul,
                          testing::ValuesIn (params));
} // namespace

