#include "Eigen/Eigen"
#include "core/command.h"
#include "ops/feed_forward.h"
#include "test_common.h"
#include "gtest/gtest.h"
#include <vector>

TEST(feed_forward, test_2d)
{
    const int indim = 1024;
    const int outdim = 1024;
    const int dim = 4 * outdim;
    const int units = (2 * dim / 3 + 31) / 32 * 32;

    std::vector<float> x(indim, .0f);
    std::vector<float> output(outdim, .0f);

    GPUDevice gpu;
    ASSERT_TRUE(gpu.init() == VK_SUCCESS) << "fail at init gpu device";
    {
        Command command(&gpu);
        ASSERT_TRUE(command.init() == VK_SUCCESS) << "fail at init command";
        ASSERT_TRUE(command.begin() == VK_SUCCESS) << "fail at begin command";

        auto w1 = random_tensor(&gpu, &command, 1, indim, units);
        auto w2 = random_tensor(&gpu, &command, 1, units, outdim);
        auto w3 = random_tensor(&gpu, &command, 1, indim, units);
        auto X = random_tensor(&gpu, &command, 1, 1, indim);

        ASSERT_TRUE(w1 && w2 && w3 && X) << "fail at creating tensors";

        FeedForward feed_forward_op(
          &gpu, &command, w1.value(), w2.value(), w3.value());

        ASSERT_TRUE(feed_forward_op.init() == VK_SUCCESS)
          << "fail at init feed_forward_op";

        VkTensor output;
        ASSERT_TRUE(feed_forward_op(X.value(), output) == VK_SUCCESS)
          << "fail at forwarding feed_forward op";

        std::vector<float> buf(outdim);
        ASSERT_TRUE(command.download(output, buf.data(), outdim) == VK_SUCCESS)
          << "fail at downloading";

        ASSERT_TRUE(command.end() == VK_SUCCESS) << "fail at end command";
        ASSERT_TRUE(command.submit_and_wait() == VK_SUCCESS)
          << "failed at submit_and_wait";
    }
}
