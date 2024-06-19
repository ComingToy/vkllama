#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/cast.h"
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
  const int dtype;
  const int heads;
  const int seqlen;
  const int maxlen;
  const int dim;
  std::array<size_t, 2> offset;
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
    ASSERT_EQ (gpu_->init (), VK_SUCCESS);
    ASSERT_EQ (command_->init (), VK_SUCCESS);
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
  ASSERT_EQ (command_->begin (), VK_SUCCESS);

  auto input0 = random_tensor<float> (gpu_, command_, params.heads,
                                      params.maxlen, params.dim);

  auto input1 = random_tensor<float> (gpu_, command_, params.heads,
                                      params.seqlen, params.dim);

  ASSERT_TRUE (input0);
  ASSERT_TRUE (input1);

  VkTensor cache, input;
  Cast cast_cache_op (gpu_, command_, VkTensor::FP32, VkTensor::FP16);
  Cast cast_input_op (gpu_, command_, VkTensor::FP32, VkTensor::FP16);

  if (params.dtype)
    {
      ASSERT_EQ (cast_cache_op.init (), VK_SUCCESS);
      ASSERT_EQ (cast_input_op.init (), VK_SUCCESS);

      ASSERT_EQ (cast_cache_op (input0->first, cache), VK_SUCCESS);
      ASSERT_EQ (cast_input_op (input1->first, input), VK_SUCCESS);
    }
  else
    {
      cache = input0->first;
      input = input1->first;
    }

  UpdateKVCache update_op (gpu_, command_, (VkTensor::DType)params.dtype);

  ASSERT_EQ (update_op.init (), VK_SUCCESS);
  ASSERT_EQ (update_op (cache, input, params.offset), VK_SUCCESS);

  std::vector<float> output_buf (cache.size ());

  VkTensor cache_fp32;
  Cast cast_output_op (gpu_, command_, VkTensor::FP16, VkTensor::FP32);
  if (params.dtype)
    {
      ASSERT_EQ (cast_output_op.init (), VK_SUCCESS);
      ASSERT_EQ (cast_output_op (cache, cache_fp32), VK_SUCCESS);
    }
  else
    {
      cache_fp32 = cache;
    }

  ASSERT_EQ (
      command_->download (cache_fp32, output_buf.data (), output_buf.size ()),
      VK_SUCCESS);
  ASSERT_EQ (command_->end (), VK_SUCCESS);
  ASSERT_EQ (command_->submit (), VK_SUCCESS);
  ASSERT_EQ (command_->wait (), VK_SUCCESS);

  Tensor<3> vk_output = TensorMap<3> (
      output_buf.data (), (Eigen::Index)cache_fp32.channels (),
      (Eigen::Index)cache_fp32.height (), (Eigen::Index)cache_fp32.width ());

  _Tensor<Eigen::half, 3> cache_eigen_tensor
      = TensorMap<3> (input0->second.data (), (Eigen::Index)cache.channels (),
                      (Eigen::Index)cache.height (),
                      (Eigen::Index)cache.width ())
            .cast<Eigen::half> ();

  _Tensor<Eigen::half, 3> input_eigen_tensor
      = TensorMap<3> (input1->second.data (), (Eigen::Index)input.channels (),
                      (Eigen::Index)input.height (),
                      (Eigen::Index)input.width ())
            .cast<Eigen::half> ();

  Eigen::array<Eigen::Index, 3> starts
      = { Eigen::Index (params.offset[0]), (Eigen::Index)params.offset[1],
          Eigen::Index (0) };

  Eigen::array<Eigen::Index, 3> sizes
      = { input_eigen_tensor.dimension (0), input_eigen_tensor.dimension (1),
          input_eigen_tensor.dimension (2) };

  cache_eigen_tensor.slice (starts, sizes) = input_eigen_tensor;

  // std::cerr << "input_tensor: " << input_eigen_tensor << std::endl
  //           << "cache: " << vk_output << std::endl
  //           << "cache eigen: " << cache_eigen_tensor << std::endl;

  Tensor<3> err (cache_eigen_tensor.dimensions ());
  err.setConstant (1e-2);

  _Tensor<int, 0> diff
      = ((cache_eigen_tensor.cast<float> () - vk_output).abs () > err)
            .cast<int> ()
            .sum ();
  ASSERT_EQ (*diff.data (), 0);
}

std::vector<TestUpdateKVCacheParams> params
    = { { 1, 32, 25, 1024, 100, { 0, 0 } } };

INSTANTIATE_TEST_SUITE_P (test_update_kv_cache, TestUpdateKVCache,
                          ::testing::ValuesIn (params));
}
