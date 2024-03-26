#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/mat_mul.h"
#include <chrono>
#include <cstdio>
#include "Eigen/Eigen"
#include "unsupported/Eigen/CXX11/Tensor"
#include <iostream>
#include <math.h>
#include <vector>
#include "test_common.h"
#include "gtest/gtest.h"

namespace {
struct TestMatMulParams
{
    const int C;
    const int M;
    const int N;
    const int K;
    const int broadcast_type;
};

class TestMatmul : public ::testing::TestWithParam<TestMatMulParams>
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

    GPUDevice* gpu_;
    Command* command_;
};

TEST_P(TestMatmul, test_matmul_broadcast)
{
    auto params = GetParam();
    const auto broadcast_type = params.broadcast_type;
    int in0_channel = 0, in1_channel = 0;

    if (broadcast_type == 0)
    {
        in0_channel = params.C;
        in1_channel = params.C;
    }
    else if (broadcast_type == 1)
    {
        in0_channel = params.C;
        in1_channel = 1;
    }
    else if (broadcast_type == 2)
    {
        in0_channel = 1;
        in1_channel = params.C;
    }
    else
    {
        ASSERT_TRUE(false) << "unsupported broadcast_type " << broadcast_type;
    }

    auto input0 = random_tensor(gpu_, command_, in0_channel, params.M, params.K);
    auto input1 = random_tensor(gpu_, command_, in1_channel, params.K, params.N);

    ASSERT_TRUE(input0 && input1) << "failed at create tensors";
    ASSERT_EQ(command_->begin(), VK_SUCCESS) << "failed at init command";

    MatMul matmul_op(gpu_, command_, 0, params.broadcast_type);

    ASSERT_TRUE(matmul_op.init() == VK_SUCCESS) << "failed at init matmul op";

    VkTensor output;
    ASSERT_TRUE(matmul_op(input0->first, input1->first, output) == VK_SUCCESS) << "failed at forwarding matmul op";

    std::vector<float> buf(output.size());
    ASSERT_EQ(command_->download(output, buf.data(), buf.size()), VK_SUCCESS) << "failed at downloading output";
    ASSERT_EQ(command_->submit_and_wait(), VK_SUCCESS) << "failed at submiting commands";

    // eigen matmul
    using TensorMap = Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor> >;
    auto in0_tensor = TensorMap(input0->second.data(), input0->first.channels(), input0->first.height(), input0->first.width());
    auto in1_tensor = TensorMap(input1->second.data(), input1->first.channels(), input1->first.height(), input1->first.width());

    Eigen::Tensor<float, 3, Eigen::RowMajor> eigen_output(output.channels(), output.height(), output.width());
    Eigen::array<Eigen::IndexPair<int>, 1> dims = {Eigen::IndexPair<int>(1, 0)};
    if (broadcast_type == 0)
    {
        for (int i = 0; i < output.channels(); ++i)
        {
            eigen_output.chip<0>(i) = in0_tensor.chip<0>(i).contract(in1_tensor.chip<0>(i), dims);
        }
    }
    else if (broadcast_type == 1)
    {
        for (int i = 0; i < output.channels(); ++i)
        {
            eigen_output.chip<0>(i) = in0_tensor.chip<0>(i).contract(in1_tensor.chip<0>(0), dims);
        }
    }
    else if (broadcast_type == 2)
    {
        for (int i = 0; i < output.channels(); ++i)
        {
            eigen_output.chip<0>(i) = in0_tensor.chip<0>(0).contract(in1_tensor.chip<0>(i), dims);
        }
    }
}
} // namespace

