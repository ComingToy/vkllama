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
  auto vocab = random_tensor<Eigen::half> (
      gpu_, command_, 1, params.VH, params.VW, Eigen::half (-1.0),
      Eigen::half (1.0), ::vkllama::DType (params.dtype));

  auto indices = random_tensor<uint32_t> (gpu_, command_, 1, params.H,
                                          params.W, 0, params.VH, UINT32);

  ASSERT_TRUE (vocab);
  ASSERT_TRUE (indices);

  Embedding emb_op (gpu_, command_, vocab->first, params.UNK,
                    (Tensor::DType)params.dtype);

  ASSERT_EQ (emb_op.init (), absl::OkStatus ());

  absl::StatusOr<Tensor> output;
  ASSERT_EQ ((output = emb_op (indices->first)).status (), absl::OkStatus ());

  std::vector<uint8_t> output_buf (output->size () * output->elem_bytes ());

  ASSERT_EQ (
      command_->download (*output, output_buf.data (), output_buf.size ()),
      absl::OkStatus ());
  ASSERT_EQ (command_->end (), absl::OkStatus ());
  ASSERT_EQ (command_->submit_and_wait (), absl::OkStatus ());

  _Tensor<Eigen::half, 2> eigen_vocab_tensor = _TensorMap<Eigen::half, 2> (
      (Eigen::half *)vocab->second.data (), params.VH, params.VW);

  _Tensor<uint32_t, 2> eigen_indices_tensor
      = _TensorMap<uint32_t, 2> (indices->second.data (), params.H, params.W);

  _Tensor<Eigen::half, 3> vk_output_tensor = _TensorMap<Eigen::half, 3> (
      (Eigen::half *)output_buf.data (), (Eigen::Index)output->channels (),
      (Eigen::Index)output->height (), (Eigen::Index)output->width ());

  _Tensor<Eigen::half, 3> eigen_output_tensor (vk_output_tensor.dimensions ());

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

  _Tensor<Eigen::half, 3> err (eigen_output_tensor.dimensions ());
  err.setConstant (Eigen::half (1e-1));

  // std::cerr << "eigen output: " << eigen_output_tensor << std::endl
  //           << "vk output: " << vk_output_tensor << std::endl;

  _Tensor<int, 0> diff
      = ((vk_output_tensor - eigen_output_tensor).abs () > err)
            .cast<int> ()
            .sum ();
  ASSERT_EQ (*diff.data (), 0);
}

std::vector<TestEmbeddingParams> params = {
  { 55, 19, 20000, 64, 0, 1 },
  { 123, 128, 20000, 64, 0, 1 },

  { 55, 19, 20000, 64, 0, 4 },
  { 123, 128, 20000, 64, 0, 4 },
};

INSTANTIATE_TEST_SUITE_P (test_embedding, TestEmbedding,
                          testing::ValuesIn (params));
}
