#include <cstdio>
#include <vector>

#include "Eigen/Eigen"
#include "core/command.h"
#include "ops/cast.h"
#include "ops/feed_forward.h"
#include "test_common.h"
#include "gtest/gtest.h"

namespace vkllama
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
    ASSERT_TRUE (command.begin () == absl::OkStatus ())
        << "fail at begin command";

    auto dtype = vkllama::DType (params.dtype);
    auto w1 = random_tensor<Eigen::half> (gpu_, &command, 1, indim, units,
                                          Eigen::half (-0.5),
                                          Eigen::half (0.5), dtype);
    auto w2 = random_tensor<Eigen::half> (gpu_, &command, 1, units, outdim,
                                          Eigen::half (-0.5),
                                          Eigen::half (0.5), dtype);
    auto w3 = random_tensor<Eigen::half> (gpu_, &command, 1, indim, units,
                                          Eigen::half (-0.5),
                                          Eigen::half (0.5), dtype);

    auto X = random_tensor<Eigen::half> (gpu_, &command, 1, 64, indim,
                                         Eigen::half (-0.5), Eigen::half (0.5),
                                         FP16);

    ASSERT_TRUE (w1 && w2 && w3 && X) << "fail at creating tensors";

    FeedForward feed_forward_op (gpu_, &command, w1->first, w2->first,
                                 w3->first, false, dtype);

    absl::Status ret;
    ASSERT_TRUE ((ret = feed_forward_op.init ()) == absl::OkStatus ())
        << "fail at init feed_forward_op: " << ret;

    auto output = feed_forward_op (X->first);
    ASSERT_TRUE (output.ok ()) << "fail at forwarding feed_forward op";

    std::vector<Eigen::half> buf (output->channels () * output->height ()
                                  * output->width ());

    ASSERT_TRUE (command.download (*output, (__vkllama_fp16_t *)buf.data (),
                                   buf.size ())
                 == absl::OkStatus ())
        << "fail at downloading";

    ASSERT_TRUE (command.end () == absl::OkStatus ()) << "fail at end command";
    ASSERT_TRUE (command.submit_and_wait () == absl::OkStatus ())
        << "failed at submit_and_wait";

    using EigenMap
        = Eigen::Map<Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor> >;

    auto ew1 = EigenMap (w1->second.data (), w1->first.height (),
                         w1->first.width ());
    auto ew2 = EigenMap (w2->second.data (), w2->first.height (),
                         w2->first.width ());
    auto ew3 = EigenMap (w3->second.data (), w3->first.height (),
                         w3->first.width ());
    auto eX
        = EigenMap (X->second.data (), X->first.height (), X->first.width ());
    auto eoutput = EigenMap (buf.data (), output->height (), output->width ());

    auto dnn1 = (eX * ew1);
    auto dnn3 = (eX * ew3);
    auto dnn1_act
        = dnn1.array () / ((-dnn1.array ()).exp () + Eigen::half (1.0f));
    auto inner = dnn1_act * dnn3.array ();
    auto output_ref = (inner.matrix () * ew2).eval ();

    auto err = Eigen::half (params.dtype ? 1e-1 : 1e-3);
    auto diff
        = ((output_ref - eoutput).array ().abs () > err).cast<int> ().sum ();
#if 0
    for (auto h = 0; h < output_ref.rows (); ++h)
      {
        for (auto w = 0; w < output_ref.cols (); ++w)
          {
            if (fabs (output_ref (h, w) - eoutput (h, w)) > err)
              {
                fprintf (stderr,
                         "index %d, %d output ref = %f, vk output = %f\n", h,
                         w, float (output_ref (h, w)), float (eoutput (h, w)));
              }
          }
      }
#endif

    ASSERT_EQ (diff, 0);
  }
}

std::vector<FeedFowardParams> params = {
  { 16, 16, 4 },   { 8, 8, 4 },     { 256, 256, 4 }, { 17, 17, 4 },
  { 16, 16, 1 },   { 8, 8, 1 },     { 256, 256, 1 }, { 17, 17, 1 },
  { 9, 9, 1 },     { 259, 259, 1 }, { 17, 19, 1 },   { 19, 17, 1 },
  { 259, 128, 1 }, { 128, 259, 1 },
};
INSTANTIATE_TEST_SUITE_P (TestFeedForawrd2DInput, TestFeedForawrd,
                          testing::ValuesIn (params));
} // namespace

