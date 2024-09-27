#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/cast.h"
#include "ops/read_kvcache_op.h"
#include "ops/update_kv_cache.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <memory>
#include <tuple>
#include <vector>

namespace vkllama
{
struct TestUpdateKVCacheParams
{
  const int heads;
  const int seqlen;
  const int maxlen;
  const int dim;
  const size_t offset;
};

class TestUpdateKVCache
    : public ::testing::TestWithParam<TestUpdateKVCacheParams>
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

TEST_P (TestUpdateKVCache, test_update_kv_cache)
{
  auto params = GetParam ();
  ASSERT_EQ (command_->begin (), absl::OkStatus ());

  auto input0 = random_tensor<Eigen::half> (gpu_, command_, params.heads,
                                            params.maxlen, params.dim);

  auto input1 = random_tensor<Eigen::half> (gpu_, command_, params.heads,
                                            params.seqlen, params.dim);

  ASSERT_TRUE (input0);
  ASSERT_TRUE (input1);

  auto cache = input0->first;
  auto input = input1->first;

  UpdateKVCache update_op (gpu_, command_, ::vkllama::FP16);
  ReadKVCache read_op (gpu_, command_);

  ASSERT_EQ (update_op.init (), absl::OkStatus ());
  ASSERT_EQ (update_op (cache, input, params.offset), absl::OkStatus ());

  absl::StatusOr<Tensor> output;
  ASSERT_EQ (read_op.init (), absl::OkStatus ());
  ASSERT_EQ (
      (output = read_op (cache, params.offset, input.height ())).status (),
      absl::OkStatus ());

  std::vector<Eigen::half> output_buf (output->size ());
  std::vector<Eigen::half> cache_buf (cache.size ());

  ASSERT_EQ (
      command_->download (*output, output_buf.data (), output_buf.size ()),
      absl::OkStatus ());
  ASSERT_EQ (command_->end (), absl::OkStatus ());
  ASSERT_EQ (command_->submit (), absl::OkStatus ());
  ASSERT_EQ (command_->wait (), absl::OkStatus ());

  _Tensor<Eigen::half, 3> vk_output = _TensorMap<Eigen::half, 3> (
      output_buf.data (), (Eigen::Index)output->channels (),
      (Eigen::Index)output->height (), (Eigen::Index)output->width ());

  // _Tensor<Eigen::half, 3> cache_eigen_tensor
  //     = TensorMap<3> (input0->second.data (), (Eigen::Index)cache.channels
  //     (),
  //                     (Eigen::Index)cache.height (),
  //                     (Eigen::Index)cache.width ())
  //           .cast<Eigen::half> ();

  _Tensor<Eigen::half, 3> input_eigen_tensor
      = _TensorMap<Eigen::half, 3> (
            input1->second.data (), (Eigen::Index)input.channels (),
            (Eigen::Index)input.height (), (Eigen::Index)input.width ())
            .cast<Eigen::half> ();

  // std::cerr << "input_tensor: " << input_eigen_tensor << std::endl
  //           << "cache: " << vk_output << std::endl
  //           << "cache eigen: " << cache_eigen_tensor << std::endl;

  _Tensor<Eigen::half, 3> err (vk_output.dimensions ());
  err.setConstant (Eigen::half (1e-2));

  _Tensor<int, 0> diff
      = ((input_eigen_tensor.cast<float> () - vk_output).abs () > err)
            .cast<int> ()
            .sum ();
  ASSERT_EQ (*diff.data (), 0);
}

std::vector<TestUpdateKVCacheParams> params
    = { { 32, 25, 1024, 100, 0 }, { 32, 25, 1024, 100, 1022 } };

INSTANTIATE_TEST_SUITE_P (test_update_kv_cache, TestUpdateKVCache,
                          ::testing::ValuesIn (params));
}
