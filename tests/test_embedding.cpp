#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/cast.h"
#include "ops/embedding.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <memory>
#include <vector>

namespace vkllama
{
struct TestEmbeddingParams
{
  uint32_t H;
  uint32_t W;
  uint32_t VH;
  uint32_t VW;
  uint32_t UNK;
  int dtype; // 0: fp32 1: fp16
};

class TestEmbedding : public ::testing::TestWithParam<TestEmbeddingParams>
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

TEST_P (TestEmbedding, test_embedding)
{
  ASSERT_EQ (command_->begin (), absl::OkStatus ());
  auto params = GetParam ();
  auto vocab = random_tensor<float> (gpu_, command_, 1, params.VH, params.VW);
  auto indices = random_tensor<uint32_t> (gpu_, command_, 1, params.H,
                                          params.W, 0, params.VH);

  ASSERT_TRUE (vocab);
  ASSERT_TRUE (indices);

  VkTensor vocab_tensor, vocab_tensor_fp16;
  Cast cast_input_op (gpu_, command_, VkTensor::FP32, VkTensor::FP16);
  Cast cast_input_op1 (gpu_, command_, VkTensor::FP16, VkTensor::FP32);

  ASSERT_EQ (cast_input_op.init (), absl::OkStatus ());
  ASSERT_EQ (cast_input_op1.init (), absl::OkStatus ());

  std::vector<float> vocab_buf (vocab->first.size ());
  if (params.dtype)
    {
      ASSERT_EQ (cast_input_op (vocab->first, vocab_tensor),
                 absl::OkStatus ());
      ASSERT_EQ (cast_input_op1 (vocab_tensor, vocab_tensor_fp16),
                 absl::OkStatus ());
      ASSERT_EQ (command_->download (vocab_tensor_fp16, vocab_buf.data (),
                                     vocab_buf.size ()),
                 absl::OkStatus ());
    }
  else
    {
      vocab_tensor = vocab->first;
      vocab_buf.swap (vocab->second);
    }

  Embedding emb_op (gpu_, command_, vocab_tensor, params.UNK,
                    (VkTensor::DType)params.dtype);

  ASSERT_EQ (emb_op.init (), absl::OkStatus ());
  VkTensor vk_output, vk_output_fp16;
  Cast cast_output_op (gpu_, command_, VkTensor::FP16, VkTensor::FP32);
  ASSERT_EQ (cast_output_op.init (), absl::OkStatus ());

  if (params.dtype)
    {
      ASSERT_EQ (emb_op (indices->first, vk_output_fp16), absl::OkStatus ());
      ASSERT_EQ (cast_output_op (vk_output_fp16, vk_output),
                 absl::OkStatus ());
    }
  else
    {
      ASSERT_EQ (emb_op (indices->first, vk_output), absl::OkStatus ());
    }

  std::vector<float> vk_output_buf (vk_output.size ());
  ASSERT_EQ (command_->download (vk_output, vk_output_buf.data (),
                                 vk_output_buf.size ()),
             absl::OkStatus ());
  ASSERT_EQ (command_->end (), absl::OkStatus ());
  ASSERT_EQ (command_->submit_and_wait (), absl::OkStatus ());

  Tensor<2> eigen_vocab_tensor
      = TensorMap<2> (vocab_buf.data (), params.VH, params.VW);
  _Tensor<uint32_t, 2> eigen_indices_tensor
      = _TensorMap<uint32_t, 2> (indices->second.data (), params.H, params.W);

  Tensor<3> vk_output_tensor = TensorMap<3> (
      vk_output_buf.data (), (Eigen::Index)vk_output.channels (),
      (Eigen::Index)vk_output.height (), (Eigen::Index)vk_output.width ());
  Tensor<3> eigen_output_tensor (vk_output_tensor.dimensions ());

  for (int i = 0; i < eigen_output_tensor.dimension (0); ++i)
    {
      for (int k = 0; k < eigen_output_tensor.dimension (1); ++k)
        {
          uint32_t tok = eigen_indices_tensor (i, k);
          if (tok >= params.VH)
            {
              tok = params.UNK;
            }

          for (int d = 0; d < eigen_vocab_tensor.dimension (1); ++d)
            {
              eigen_output_tensor (i, k, d) = eigen_vocab_tensor (tok, d);
            }
        }
    }

#if 0
  for (auto i = 0; i < eigen_output_tensor.size (); ++i)
    {
      if (fabs (eigen_output_tensor (i) - vk_output_tensor (i)) <= 0.01)
        {
          continue;
        }
      fprintf (stderr,
               "index %d eigen output value: %f, vk output value: %f\n", i,
               eigen_output_tensor (i), vk_output_tensor (i));
    }
#endif

  Tensor<3> err (eigen_output_tensor.dimensions ());
  err.setConstant (1e-2);

  _Tensor<int, 0> diff
      = ((vk_output_tensor - eigen_output_tensor).abs () > err)
            .cast<int> ()
            .sum ();
  ASSERT_EQ (*diff.data (), 0);
}

std::vector<TestEmbeddingParams> params = {
  { 55, 19, 20000, 64, 0, 0 },
  { 123, 128, 20000, 64, 0, 0 },
  { 55, 19, 20000, 64, 0, 1 },
  { 123, 128, 20000, 64, 0, 1 },

};

INSTANTIATE_TEST_SUITE_P (test_embedding, TestEmbedding,
                          testing::ValuesIn (params));
}
