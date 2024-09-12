#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/cast.h"
#include "ops/concat.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <core/float.h>
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
  std::vector<Tensor> input_tensors;
  std::vector<std::vector<__vkllama_fp16_t> > input_bufs;

  for (auto const &[c, h, w] : params.shapes)
    {
      auto input = random_tensor<__vkllama_fp16_t> (gpu_, command_, c, h, w,
                                                    __fp32_to_fp16 (-1.0),
                                                    __fp32_to_fp16 (1.0));

      ASSERT_TRUE (input) << "failed at create tensor";
      input_tensors.push_back (input->first);
      input_bufs.push_back (input->second);
    }

  Concat concat_op (gpu_, command_, (int)input_tensors.size (), params.axis,
                    (Tensor::DType)params.dtype);
  ASSERT_EQ (concat_op.init (), absl::OkStatus ()) << "failed at init op";

  absl::StatusOr<Tensor> output;
  ASSERT_EQ ((output = concat_op (input_tensors)).status (),
             absl::OkStatus ());

  std::vector<uint8_t> output_buf (output->size () * output->elem_bytes ());

  ASSERT_EQ (
      command_->download (*output, output_buf.data (), output_buf.size ()),
      absl::OkStatus ())
      << "failed at download output";

  ASSERT_EQ (command_->end (), absl::OkStatus ()) << "failed at end commands";
  ASSERT_EQ (command_->submit_and_wait (), absl::OkStatus ())
      << "failed at submit commands";

  std::vector<_Tensor<Eigen::half, 3> > input_eigen_tensors (
      input_tensors.size ());
  for (int i = 0; i < input_eigen_tensors.size (); ++i)
    {
      auto const &t = input_tensors[i];
      input_eigen_tensors[i] = _TensorMap<Eigen::half, 3> (
          (Eigen::half *)input_bufs[i].data (), (Eigen::Index)t.channels (),
          (Eigen::Index)t.height (), (Eigen::Index)t.width ());
    }

  _Tensor<Eigen::half, 3> tmp = input_eigen_tensors[0];
  _Tensor<Eigen::half, 3> eigen_output_tensor;

  for (int i = 1; i < input_eigen_tensors.size (); ++i)
    {
      eigen_output_tensor
          = tmp.concatenate (input_eigen_tensors[i], params.axis);
      tmp = eigen_output_tensor;
    }

  // _Tensor<float, 3> eigen_output_tensor = concat_expr;

  _Tensor<Eigen::half, 3> vk_output_tensor = _TensorMap<Eigen::half, 3> (
      (Eigen::half *)output_buf.data (), (Eigen::Index)output->channels (),
      (Eigen::Index)output->height (), (Eigen::Index)output->width ());

  _Tensor<Eigen::half, 3> err (vk_output_tensor.dimensions ());
  err.setConstant (params.dtype ? Eigen::half (1e-2) : Eigen::half (1e-3));
  _Tensor<int, 0> diff
      = ((vk_output_tensor - eigen_output_tensor).abs () > err)
            .cast<int> ()
            .sum ();
  ASSERT_EQ (*diff.data (), 0);
}

std::vector<TestConcatParams> params = {
  { 1, 0, { { 1, 63, 31 }, { 2, 63, 31 }, { 5, 63, 31 } } },
  { 1, 1, { { 3, 63, 31 }, { 3, 31, 31 }, { 3, 22, 31 } } },
  { 1, 2, { { 3, 63, 31 }, { 3, 63, 63 }, { 3, 63, 22 } } },
};

INSTANTIATE_TEST_SUITE_P (test_concat, TestConcat,
                          ::testing::ValuesIn (params));
}
