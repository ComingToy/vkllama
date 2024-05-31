#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/cast.h"
#include "ops/rope.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <vector>

namespace
{
struct TestRopeParams
{
  const int C;
  const int H;
  const int W;
  const int MAXLEN;
  const int dtype;
};

class TestRope : public ::testing::TestWithParam<TestRopeParams>
{
public:
  void
  SetUp () override
  {
    dev_ = new GPUDevice ();
    command_ = new Command (dev_);

    dev_->init ();
    command_->init ();
  }
  void
  TearDown () override
  {
    delete command_;
    delete dev_;
  }

  GPUDevice *dev_;
  Command *command_;
};

TEST_P (TestRope, test_rope)
{
  auto params = GetParam ();

  ASSERT_EQ (command_->begin (), VK_SUCCESS) << "failed at beign command";
  auto input_query
      = random_tensor<float> (dev_, command_, params.C, params.H, params.W);
  auto input_key
      = random_tensor<float> (dev_, command_, params.C, params.H, params.W);

  ASSERT_TRUE (input_query && input_key);
  VkTensor input_query_fp16, input_query_fp32, input_key_fp16, input_key_fp32;
  std::vector<float> input_query_buf (input_query->first.size ()),
      input_key_buf (input_key->first.size ());

  Cast input_query_cast_fp16 (dev_, command_, VkTensor::FP32, VkTensor::FP16);
  Cast input_key_cast_fp16 (dev_, command_, VkTensor::FP32, VkTensor::FP16);
  Cast input_query_cast_fp32 (dev_, command_, VkTensor::FP16, VkTensor::FP32);
  Cast input_key_cast_fp32 (dev_, command_, VkTensor::FP16, VkTensor::FP32);
  if (params.dtype)
    {
      ASSERT_EQ (input_query_cast_fp16.init (), VK_SUCCESS);
      ASSERT_EQ (input_key_cast_fp16.init (), VK_SUCCESS);
      ASSERT_EQ (input_query_cast_fp32.init (), VK_SUCCESS);
      ASSERT_EQ (input_key_cast_fp32.init (), VK_SUCCESS);

      ASSERT_EQ (input_query_cast_fp16 (input_query->first, input_query_fp16),
                 VK_SUCCESS);
      ASSERT_EQ (input_key_cast_fp16 (input_key->first, input_key_fp16),
                 VK_SUCCESS);
      ASSERT_EQ (input_query_cast_fp32 (input_query_fp16, input_query_fp32),
                 VK_SUCCESS);
      ASSERT_EQ (input_key_cast_fp32 (input_key_fp16, input_key_fp32),
                 VK_SUCCESS);
      ASSERT_EQ (command_->download (input_key_fp32, input_key_buf.data (),
                                     input_key_buf.size ()),
                 VK_SUCCESS);
      ASSERT_EQ (command_->download (input_query_fp32, input_query_buf.data (),
                                     input_query_buf.size ()),
                 VK_SUCCESS);
    }
  else
    {
      input_query_fp32 = input_query->first;
      input_key_fp32 = input_key->first;
      input_query_buf.swap (input_query->second);
      input_key_buf.swap (input_key->second);
    }

  Rope rope_op (dev_, command_, params.MAXLEN, params.W,
                (VkTensor::DType)params.dtype);

  ASSERT_EQ (rope_op.init (), VK_SUCCESS);

  VkTensor output_query, output_key;
  ASSERT_EQ (rope_op (params.dtype ? input_query_fp16 : input_query_fp32,
                      params.dtype ? input_key_fp16 : input_key_fp32,
                      output_query, output_key),
             VK_SUCCESS);

  std::vector<float> output_query_buf (output_query.size ()),
      output_key_buf (output_key.size ());

  Cast cast_output_query_op (dev_, command_, VkTensor::FP16, VkTensor::FP32);
  Cast cast_output_key_op (dev_, command_, VkTensor::FP16, VkTensor::FP32);

  VkTensor output_query_fp32, output_key_fp32;
  if (params.dtype)
    {
      ASSERT_EQ (cast_output_key_op.init (), VK_SUCCESS);
      ASSERT_EQ (cast_output_query_op.init (), VK_SUCCESS);

      ASSERT_EQ (cast_output_key_op (output_key, output_key_fp32), VK_SUCCESS);
      ASSERT_EQ (cast_output_query_op (output_query, output_query_fp32),
                 VK_SUCCESS);
    }
  else
    {
      output_key_fp32 = output_key;
      output_query_fp32 = output_query;
    }

  ASSERT_EQ (command_->download (output_query_fp32, output_query_buf.data (),
                                 output_query_buf.size ()),
             VK_SUCCESS);
  ASSERT_EQ (command_->download (output_key_fp32, output_key_buf.data (),
                                 output_key.size ()),
             VK_SUCCESS);
  ASSERT_EQ (command_->end (), VK_SUCCESS) << "failed at end commands";
  ASSERT_EQ (command_->submit_and_wait (), VK_SUCCESS)
      << "failed at submit commands";

  auto output_query_host = TensorMap<3> (
      output_query_buf.data (), (Eigen::Index)output_query.channels (),
      (Eigen::Index)output_query.height (),
      (Eigen::Index)output_query.width ());
  auto output_key_host = TensorMap<3> (
      output_key_buf.data (), (Eigen::Index)output_key.channels (),
      (Eigen::Index)output_key.height (), (Eigen::Index)output_key.width ());

  auto input_query_host = TensorMap<3> (
      input_query_buf.data (), (Eigen::Index)input_query->first.channels (),
      (Eigen::Index)input_query->first.height (),
      (Eigen::Index)input_query->first.width ());

  auto input_key_host = TensorMap<3> (
      input_key_buf.data (), (Eigen::Index)input_key->first.channels (),
      (Eigen::Index)input_key->first.height (),
      (Eigen::Index)input_key->first.width ());
  size_t w = input_query->first.width () / 2;

  auto freqc_buf = rope_op.freqc ();
  auto freqs_buf = rope_op.freqs ();
  auto input_freqc_host = TensorMap<3> (
      freqc_buf.data (), (Eigen::Index)input_query->first.channels (),
      (Eigen::Index)input_query->first.height (), (Eigen::Index)w);
  auto input_freqs_host = TensorMap<3> (
      freqs_buf.data (), (Eigen::Index)input_query->first.channels (),
      (Eigen::Index)input_query->first.height (), (Eigen::Index)w);

  auto _apply_rope
      = [] (Tensor<3> input_x, Tensor<3> input_freqc, Tensor<3> input_freqs)

  {
    // apply rope to query
    Eigen::array<Eigen::Index, 4> dims
        = { input_x.dimension (0), input_x.dimension (1),
            input_x.dimension (2) / 2, (Eigen::Index)2 };
    Tensor<4> query_host = input_x.reshape (dims);
    Tensor<3> query_host_r = query_host.chip<3> (0);
    Tensor<3> query_host_i = query_host.chip<3> (1);

    Tensor<3> query_host_or (query_host_r.dimensions ());
    Tensor<3> query_host_oi (query_host_r.dimensions ());
    for (int i = 0; i < query_host_or.dimension (0); ++i)
      {
        query_host_or.chip<0> (i)
            = query_host_r.chip<0> (i) * input_freqc.chip<0> (0)
              - query_host_i.chip<0> (i) * input_freqs.chip<0> (0);
        query_host_oi.chip<0> (i)
            = query_host_r.chip<0> (i) * input_freqs.chip<0> (0)
              + query_host_i.chip<0> (i) * input_freqc.chip<0> (0);
      }

    // auto output_query_host_ref =
    // query_host_or.reshape(Eigen::array<Eigen::Index,
    // 4>{query_host_or.dimension(0), query_host_or.dimension(1),
    // query_host_or.dimension(2),
    // (Eigen::Index)1}).concatenate(query_host_oi.reshape(Eigen::array<Eigen::Index,
    // 4>{query_host_oi.dimension(0), query_host_oi.dimension(1),
    // query_host_oi.dimension(2), (Eigen::Index)1}), 3);
    Eigen::array<Eigen::Index, 4> out_dims
        = { query_host_or.dimension (0), query_host_or.dimension (1),
            query_host_or.dimension (2), (Eigen::Index)1 };
    Tensor<4> reshaped_query_host_or = query_host_or.reshape (out_dims);
    Tensor<4> reshaped_query_host_oi = query_host_oi.reshape (out_dims);
    Tensor<4> output
        = reshaped_query_host_or.concatenate (reshaped_query_host_oi, 3);

    Tensor<3> output_ = output.reshape (input_x.dimensions ());
    return output_;
  };

  auto rope_output_query
      = _apply_rope (input_query_host, input_freqc_host, input_freqs_host);
  auto rope_output_key
      = _apply_rope (input_key_host, input_freqc_host, input_freqs_host);
  auto rope_vulkan_output_query = TensorMap<3> (
      output_query_buf.data (), input_query_host.dimensions ());
  auto rope_vulkan_output_key
      = TensorMap<3> (output_key_buf.data (), input_query_host.dimensions ());

  Tensor<3> err (rope_output_query.dimensions ());
  err.setConstant (params.dtype ? 1e-2 : 1e-3);

  _Tensor<int, 0> diff
      = ((rope_output_key - rope_vulkan_output_key).abs () > err)
            .cast<int> ()
            .sum ();
  ASSERT_EQ (*diff.data (), 0);

  _Tensor<int, 0> diff_q
      = ((rope_output_query - rope_vulkan_output_query).abs () > err)
            .cast<int> ()
            .sum ();
  ASSERT_EQ (*diff_q.data (), 0);
}

std::vector<TestRopeParams> params = {
  { 1, 256, 64, 256, 0 },  { 1, 128, 64, 256, 0 },
  { 12, 256, 64, 256, 0 }, { 15, 128, 64, 256, 0 },

  { 1, 256, 64, 256, 1 },  { 1, 128, 64, 256, 1 },
  { 12, 256, 64, 256, 1 }, { 15, 128, 64, 256, 1 },
};

INSTANTIATE_TEST_SUITE_P (TestRopeCases, TestRope, testing::ValuesIn (params));
} // namespace
