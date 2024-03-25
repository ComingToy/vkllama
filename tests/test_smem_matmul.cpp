#include "core/allocator.h"
#include "core/command.h"
#include "core/gpu_device.h"
#include "core/pipeline.h"
#include "ops/mat_mul.h"
#include <chrono>
#include <cstdio>
#include "Eigen/Eigen"
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

class TestMatMulParams : public ::testing::TestWithParam<TestMatMulParams>
{
};
} // namespace

Eigen::Map<
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >
    A(val1, M, K);
Eigen::Map<
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >
    B(val2, K, N);
Eigen::Map<
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >
    C(val3, M, N);

int main(const int argc, const char* argv[])
{
    random_vec(val1, M * K);
    random_vec(val2, K * M);
    random_vec(val3, M * N);

    GPUDevice gpu;
    if (gpu.init() != VK_SUCCESS)
    {
        fprintf(stderr, "init gpu failed.\n");
        return -1;
    }

    {
        VkTensor b1(1, M, K, &gpu, true);
        VkTensor b2(1, K, N, &gpu, true);
        VkTensor b3;

        if (b1.create() != VK_SUCCESS || b2.create() != VK_SUCCESS)
        {
            return -1;
        }

        Command command(&gpu);
        auto ret = command.init();
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "failed at init command\n");
            return -1;
        }

        command.begin();
        command.upload(val1, M * K, b1);
        command.upload(val2, K * N, b2);

        MatMul matmul(&gpu, &command);
        if (matmul.init() != VK_SUCCESS)
        {
            fprintf(stderr, "failed at init op\n");
            return -1;
        }
        ret = matmul(b1, b2, b3);

        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "failed at matmul operator\n");
            return -1;
        }
        command.download(b3, val3, M * N);
        command.end();
        command.submit_and_wait();

        std::cerr << "time cost: " << matmul.time() << std::endl;
        auto c = A * B;
        auto mse = (C - c).array().pow(2.f).mean();
        std::cerr << "mse: " << mse << std::endl;
    }
    return 0;
}
