#include <cstdio>
#include <vector>

#include "Eigen/Eigen"
#include "core/command.h"
#include "ops/cast.h"
#include "ops/feed_forward.h"
#include "test_common.h"
#include "gtest/gtest.h"

namespace
{
struct FeedFowardParams
{
  int indim;
  int outdim;
  int dtype;
};

class TestFeedForawrd : public testing::TestWithParam<FeedFowardParams>
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

public:
  GPUDevice *gpu_;
  Command *command_;
};

TEST_P (TestFeedForawrd, test_2d)
{
  auto params = GetParam ();
  auto indim = params.indim;
  auto outdim = params.outdim;

  const int dim = 4 * outdim;
  const int units = (2 * dim / 3 + 31) / 32 * 32;

  std::vector<float> x (indim, .0f);
  std::vector<float> output (outdim, .0f);

  auto &command = *command_;
  {
    ASSERT_TRUE (command.begin () == VK_SUCCESS) << "fail at begin command";

    auto w1 = random_tensor<float> (gpu_, &command, 1, indim, units);
    auto w2 = random_tensor<float> (gpu_, &command, 1, units, outdim);
    auto w3 = random_tensor<float> (gpu_, &command, 1, indim, units);
    auto X = random_tensor<float> (gpu_, &command, 1, 64, indim);
    ASSERT_TRUE (w1 && w2 && w3 && X) << "fail at creating tensors";

    VkTensor w1_fp16, w2_fp16, w3_fp16, X_fp16, w1_fp32, w2_fp32, w3_fp32,
        X_fp32;

    std::vector<float> w1_buf (w1->second.size ()),
        w2_buf (w2->second.size ()), w3_buf (w3->second.size ()),
        X_buf (X->second.size ());

    Cast cast_w1_op (gpu_, command_, VkTensor::FP32, VkTensor::FP16);
    Cast cast_w2_op (gpu_, command_, VkTensor::FP32, VkTensor::FP16);
    Cast cast_w3_op (gpu_, command_, VkTensor::FP32, VkTensor::FP16);
    Cast cast_X_op (gpu_, command_, VkTensor::FP32, VkTensor::FP16);

    Cast cast_w1_op1 (gpu_, command_, VkTensor::FP16, VkTensor::FP32);
    Cast cast_w2_op1 (gpu_, command_, VkTensor::FP16, VkTensor::FP32);
    Cast cast_w3_op1 (gpu_, command_, VkTensor::FP16, VkTensor::FP32);
    Cast cast_X_op1 (gpu_, command_, VkTensor::FP16, VkTensor::FP32);

    if (params.dtype)
      {
        ASSERT_EQ (cast_w1_op.init (), VK_SUCCESS);
        ASSERT_EQ (cast_w2_op.init (), VK_SUCCESS);
        ASSERT_EQ (cast_w3_op.init (), VK_SUCCESS);
        ASSERT_EQ (cast_X_op.init (), VK_SUCCESS);

        ASSERT_EQ (cast_w1_op1.init (), VK_SUCCESS);
        ASSERT_EQ (cast_w2_op1.init (), VK_SUCCESS);
        ASSERT_EQ (cast_w3_op1.init (), VK_SUCCESS);
        ASSERT_EQ (cast_X_op1.init (), VK_SUCCESS);

        ASSERT_EQ (cast_w1_op (w1->first, w1_fp16), VK_SUCCESS);
        ASSERT_EQ (cast_w2_op (w2->first, w2_fp16), VK_SUCCESS);
        ASSERT_EQ (cast_w3_op (w3->first, w3_fp16), VK_SUCCESS);
        ASSERT_EQ (cast_X_op (X->first, X_fp16), VK_SUCCESS);

        ASSERT_EQ (cast_w1_op1 (w1_fp16, w1_fp32), VK_SUCCESS);
        ASSERT_EQ (cast_w2_op1 (w2_fp16, w2_fp32), VK_SUCCESS);
        ASSERT_EQ (cast_w3_op1 (w3_fp16, w3_fp32), VK_SUCCESS);
        ASSERT_EQ (cast_X_op1 (X_fp16, X_fp32), VK_SUCCESS);

        ASSERT_EQ (
            command_->download (w1_fp32, w1_buf.data (), w1_buf.size ()),
            VK_SUCCESS);

        ASSERT_EQ (
            command_->download (w2_fp32, w2_buf.data (), w2_buf.size ()),
            VK_SUCCESS);

        ASSERT_EQ (
            command_->download (w3_fp32, w3_buf.data (), w3_buf.size ()),
            VK_SUCCESS);

        ASSERT_EQ (command_->download (X_fp32, X_buf.data (), X_buf.size ()),
                   VK_SUCCESS);
      }
    else
      {
        w1_buf.swap (w1->second);
        w2_buf.swap (w2->second);
        w3_buf.swap (w3->second);
        X_buf.swap (X->second);
        w1_fp32 = w1->first;
        w2_fp32 = w2->first;
        w3_fp32 = w3->first;
        X_fp32 = X->first;
      }

    VkTensor::DType dtype = (VkTensor::DType)params.dtype;
    FeedForward feed_forward_op (
        gpu_, &command, dtype == VkTensor::FP32 ? w1_fp32 : w1_fp16,
        dtype == VkTensor::FP32 ? w2_fp32 : w2_fp16,
        dtype == VkTensor::FP32 ? w3_fp32 : w3_fp16, false, dtype);

    ASSERT_TRUE (feed_forward_op.init () == VK_SUCCESS)
        << "fail at init feed_forward_op";

    VkTensor output;
    ASSERT_TRUE (
        feed_forward_op (dtype == VkTensor::FP32 ? X_fp32 : X_fp16, output)
        == VK_SUCCESS)
        << "fail at forwarding feed_forward op";

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

    std::vector<float> buf (output.channels () * output.height ()
                            * output.width ());

    ASSERT_TRUE (command.download (output_fp32, buf.data (), buf.size ())
                 == VK_SUCCESS)
        << "fail at downloading";

    ASSERT_TRUE (command.end () == VK_SUCCESS) << "fail at end command";
    ASSERT_TRUE (command.submit_and_wait () == VK_SUCCESS)
        << "failed at submit_and_wait";

    using EigenMap
        = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor> >;

    auto ew1
        = EigenMap (w1_buf.data (), w1->first.height (), w1->first.width ());
    auto ew2
        = EigenMap (w2_buf.data (), w2->first.height (), w2->first.width ());
    auto ew3
        = EigenMap (w3_buf.data (), w3->first.height (), w3->first.width ());
    auto eX = EigenMap (X_buf.data (), X->first.height (), X->first.width ());
    auto eoutput = EigenMap (buf.data (), output.height (), output.width ());

    auto dnn1 = (eX * ew1);
    auto dnn3 = (eX * ew3);
    auto dnn1_act = dnn1.array () / ((-dnn1.array ()).exp () + 1.0f);
    auto inner = dnn1_act * dnn3.array ();
    auto output_ref = inner.matrix () * ew2;

    auto mse = (output_ref.array () - eoutput.array ()).pow (2.0f).mean ();
    ASSERT_LT (mse, 1e-4);
  }
}

std::vector<FeedFowardParams> params = {
#if 0
  { 16, 16, 0 },   { 8, 8, 0 },     { 256, 256, 0 }, { 17, 17, 0 },
  { 9, 9, 0 },     { 259, 259, 0 }, { 17, 19, 0 },   { 19, 17, 0 },
  { 259, 128, 0 }, { 128, 259, 0 },
#else
  { 16, 16, 1 },   { 8, 8, 1 },     { 256, 256, 1 }, { 17, 17, 1 },
  { 9, 9, 1 },     { 259, 259, 1 }, { 17, 19, 1 },   { 19, 17, 1 },
  { 259, 128, 1 }, { 128, 259, 1 },
#endif
};
INSTANTIATE_TEST_SUITE_P (TestFeedForawrd2DInput, TestFeedForawrd,
                          testing::ValuesIn (params));
} // namespace

