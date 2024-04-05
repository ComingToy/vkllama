#include "core/command.h"
#include "core/gpu_device.h"
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
      = random_tensor (dev_, command_, params.C, params.H, params.W);
  auto input_key
      = random_tensor (dev_, command_, params.C, params.H, params.W);

  ASSERT_TRUE (input_query && input_key);

  Rope rope_op (dev_, command_, params.MAXLEN, params.W);
  ASSERT_EQ (rope_op.init (), VK_SUCCESS);

  VkTensor output_query, output_key;
  ASSERT_EQ (
      rope_op (input_query->first, input_key->first, output_query, output_key),
      VK_SUCCESS);

  std::vector<float> output_query_buf (output_query.size ()),
      output_key_buf (output_key.size ());
  ASSERT_EQ (command_->download (output_query, output_query_buf.data (),
                                 output_query_buf.size ()),
             VK_SUCCESS);
  ASSERT_EQ (command_->download (output_key, output_key_buf.data (),
                                 output_key.size ()),
             VK_SUCCESS);
  ASSERT_EQ (command_->end (), VK_SUCCESS) << "failed at end commands";
  ASSERT_EQ (command_->submit_and_wait (), VK_SUCCESS)
      << "failed at submit commands";

  auto output_query_host
      = TensorMap<3> (output_query_buf.data (), output_query.channels (),
                      output_query.height (), output_query.width ());
  auto output_key_host
      = TensorMap<3> (output_key_buf.data (), output_key.channels (),
                      output_key.height (), output_key.width ());

  auto input_query_host = TensorMap<3> (
      input_query->second.data (), input_query->first.channels (),
      input_query->first.height (), input_query->first.width ());
  auto input_key_host
      = TensorMap<3> (input_key->second.data (), input_key->first.channels (),
                      input_key->first.height (), input_key->first.width ());
  size_t w = input_query->first.width () / 2;

  auto freqc_buf = rope_op.freqc ();
  auto freqs_buf = rope_op.freqs ();
  auto input_freqc_host
      = TensorMap<3> (freqc_buf.data (), input_query->first.channels (),
                      input_query->first.height (), w);
  auto input_freqs_host
      = TensorMap<3> (freqs_buf.data (), input_query->first.channels (),
                      input_query->first.height (), w);

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

  Tensor<0> mse0
      = (rope_output_query - rope_vulkan_output_query).pow (2.0f).mean ();
  Tensor<0> mse1
      = (rope_output_key - rope_vulkan_output_key).pow (2.0f).mean ();
  ASSERT_LT (*mse0.data (), 1e-4);
  ASSERT_LT (*mse1.data (), 1e-4);
}

std::vector<TestRopeParams> params = {
  { 1, 256, 64, 256 },
  { 1, 128, 64, 256 },
  { 12, 256, 64, 256 },
  { 15, 128, 64, 256 },
};

INSTANTIATE_TEST_SUITE_P (TestRopeCases, TestRope, testing::ValuesIn (params));
} // namespace
