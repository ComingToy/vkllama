#include <cstdio>
#include <vector>

#include "Eigen/Eigen"
#include "core/command.h"
#include "gtest/gtest.h"
#include "ops/feed_forward.h"
#include "test_common.h"

namespace {
struct FeedFowardParams
{
    int indim;
    int outdim;
};

class TestFeedForawrd : public testing::TestWithParam<FeedFowardParams>
{
public:
    void SetUp() override
    {
        gpu_ = new GPUDevice();
        gpu_->init();
        command_ = new Command(gpu_);
        command_->init();
    };

    void TearDown() override
    {
        delete command_;
        delete gpu_;
    }

public:
    GPUDevice* gpu_;
    Command* command_;
};

TEST_P(TestFeedForawrd, test_2d)
{
    auto params = GetParam();
    auto indim = params.indim;
    auto outdim = params.outdim;

    const int dim = 4 * outdim;
    const int units = (2 * dim / 3 + 31) / 32 * 32;

    std::vector<float> x(indim, .0f);
    std::vector<float> output(outdim, .0f);

    auto& command = *command_;
    {
        ASSERT_TRUE(command.begin() == VK_SUCCESS) << "fail at begin command";

        auto w1 = random_tensor(gpu_, &command, 1, indim, units);
        auto w2 = random_tensor(gpu_, &command, 1, units, outdim);
        auto w3 = random_tensor(gpu_, &command, 1, indim, units);
        auto X = random_tensor(gpu_, &command, 1, 64, indim);

        ASSERT_TRUE(w1 && w2 && w3 && X) << "fail at creating tensors";

        FeedForward feed_forward_op(gpu_, &command, w1.value().first,
                                    w2.value().first, w3.value().first);

        ASSERT_TRUE(feed_forward_op.init() == VK_SUCCESS)
            << "fail at init feed_forward_op";

        VkTensor output;
        ASSERT_TRUE(feed_forward_op(X.value().first, output) == VK_SUCCESS)
            << "fail at forwarding feed_forward op";

        std::vector<float> buf(output.channels() * output.height() * output.width());
        ASSERT_TRUE(command.download(output, buf.data(), buf.size()) == VK_SUCCESS)
            << "fail at downloading";

        ASSERT_TRUE(command.end() == VK_SUCCESS) << "fail at end command";
        ASSERT_TRUE(command.submit_and_wait() == VK_SUCCESS)
            << "failed at submit_and_wait";

        using EigenMap = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                                  Eigen::RowMajor> >;
        auto ew1 = EigenMap(w1.value().second.data(), w1.value().first.height(), w1.value().first.width());
        auto ew2 = EigenMap(w2.value().second.data(), w2.value().first.height(), w2.value().first.width());
        auto ew3 = EigenMap(w3.value().second.data(), w3.value().first.height(), w3.value().first.width());
        auto eX = EigenMap(X.value().second.data(), X.value().first.height(), X.value().first.width());
        auto eoutput = EigenMap(buf.data(), output.height(), output.width());

        auto dnn1 = (eX * ew1);
        auto dnn3 = (eX * ew3);
        auto dnn1_act = dnn1.array() / ((-dnn1.array()).exp() + 1.0f);
        auto inner = dnn1_act * dnn3.array();
        auto output_ref = inner.matrix() * ew2;

        auto mse = (output_ref.array() - eoutput.array()).pow(2.0f).mean();
        ASSERT_LT(mse, 1e-4);
    }
}

std::vector<FeedFowardParams> params = {
    {16, 16},
    {8, 8},
    {256, 256},
    {17, 17},
    {9, 9},
    {259, 259},
    {17, 19},
    {19, 17},
    {259, 128},
    {128, 259},

    {16, 16},
    {8, 8},
    {256, 256},
    {17, 17},
    {9, 9},
    {259, 259},
    {17, 19},
    {19, 17},
    {259, 128},
    {128, 259},

};
INSTANTIATE_TEST_SUITE_P(TestFeedForawrd2DInput, TestFeedForawrd,
                         testing::ValuesIn(params));
} // namespace

