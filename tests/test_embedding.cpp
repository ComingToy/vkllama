#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/embedding.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>

namespace
{
struct TestEmbeddingParams
{
  uint32_t H;
  uint32_t W;
  uint32_t VH;
  uint32_t VW;
  uint32_t UNK;
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
  ASSERT_EQ (command_->begin (), VK_SUCCESS);
  auto params = GetParam ();
  auto vocab = random_tensor<float> (gpu_, command_, 1, params.VH, params.VW);
  auto indices = random_tensor<uint32_t> (gpu_, command_, 1, params.H,
                                          params.W, 0, params.VH);

  ASSERT_TRUE (vocab);
  ASSERT_TRUE (indices);

  Embedding emb_op (gpu_, command_, params.UNK);
  ASSERT_EQ (emb_op.init (), VK_SUCCESS);
  VkTensor vk_output;

  ASSERT_EQ (emb_op (vocab->first, indices->first, vk_output), VK_SUCCESS);
  std::vector<float> vk_output_buf (vk_output.size ());
  ASSERT_EQ (command_->download (vk_output, vk_output_buf.data (),
                                 vk_output_buf.size ()),
             VK_SUCCESS);
  ASSERT_EQ (command_->end (), VK_SUCCESS);
  ASSERT_EQ (command_->submit_and_wait (), VK_SUCCESS);

  Tensor<2> eigen_vocab_tensor
      = TensorMap<2> (vocab->second.data (), params.VH, params.VW);
  _Tensor<uint32_t, 2> eigen_indices_tensor
      = _TensorMap<uint32_t, 2> (indices->second.data (), params.H, params.W);

  Tensor<3> vk_output_tensor
      = TensorMap<3> (vk_output_buf.data (), vk_output.channels (),
                      vk_output.height (), vk_output.width ());
  Tensor<3> eigen_output_tensor (vk_output_tensor.dimensions ());

  for (int i = 0; i < eigen_output_tensor.dimension (0); ++i)
    {
      for (int k = 0; k < eigen_output_tensor.dimension (1); ++k)
        {
          uint32_t tok = eigen_indices_tensor (i, k);
          for (int d = 0; d < eigen_vocab_tensor.dimension (1); ++d)
            {
              eigen_output_tensor (i, k, d) = eigen_vocab_tensor (tok, d);
            }
        }
    }

  Tensor<3> err (eigen_output_tensor.dimensions ());
  err.setConstant (1e-3);

  _Tensor<int, 0> diff
      = ((vk_output_tensor - eigen_output_tensor).abs () > err)
            .cast<int> ()
            .sum ();
  ASSERT_EQ (*diff.data (), 0);
}

std::vector<TestEmbeddingParams> params
    = { { 55, 19, 20000, 64, 0 }, { 123, 128, 20000, 64, 0 } };

INSTANTIATE_TEST_SUITE_P (test_embedding, TestEmbedding,
                          testing::ValuesIn (params));
}
