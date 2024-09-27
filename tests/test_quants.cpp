#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "core/quants.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <memory>
#include <tuple>
#include <vector>

namespace vkllama
{
struct TestQuantsParams
{
  const size_t n;
  const int dtype;
};

using TestQuants = VkllamaTestWithParam<TestQuantsParams>;

TEST_P (TestQuants, test_quants)
{
  auto params = GetParam ();
  std::vector<float> buf (params.n);

  random_vec (buf.data (), buf.size (), -5.0f, 5.0f);

  std::vector<int8_t> q8_0_buf (((buf.size () + 31) / 32) * 36);

  absl::Status ret;
  ret = qint8_0_quantize (buf.data (), q8_0_buf.data (), 1, buf.size ());
  ASSERT_TRUE (ret.ok ()) << ret;

  std::vector<float> de_q8_0_buf;
  de_q8_0_buf.resize (buf.size ());

  ret = vkllama::qint8_0_dequantize (q8_0_buf.data (), de_q8_0_buf.data (), 1,
                                     buf.size ());

  ASSERT_TRUE (ret.ok ()) << ret;

  auto raw = _TensorMap<float, 2> (buf.data (), 1, (Eigen::Index)buf.size ());
  auto de_q8_0 = _TensorMap<float, 2> (de_q8_0_buf.data (), 1,
                                       (Eigen::Index)de_q8_0_buf.size ());

  std::cerr << "raw data: " << raw << std::endl
            << "dequantized data: " << de_q8_0 << std::endl;
  _Tensor<float, 2> err (1, buf.size ());
  err.setConstant (Eigen::half (1e-1));

  _Tensor<int, 0> diff = ((raw - de_q8_0).abs () > err).cast<int> ().sum ();
  ASSERT_EQ (*diff.data (), 0);
}

std::vector<TestQuantsParams> params = {
#if 0
		{ 1024, 2 }, { 2048, 2 },
#endif
  { 95, 2 }
};

INSTANTIATE_TEST_SUITE_P (test_quants, TestQuants,
                          ::testing::ValuesIn (params));
}
