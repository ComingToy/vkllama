#include "shaders/vkllama_shaders.h"
#include "src/core/allocator.h"
#include "src/core/command.h"
#include "src/core/gpu_device.h"
#include "src/core/pipeline.h"
#include "src/ops/mat_mul.h"
#include <chrono>
#include <cstdio>
#include "Eigen/Eigen"
#include <iostream>
#include <math.h>
#include <vector>

void
random_vec(float* v, const int n)
{
    for (int i = 0; i < n; ++i) {
        v[i] = static_cast<float>(random() % 100) / 50.0f;
    }
}

void
clear_vec(float* v, const int n)
{
    for (int i = 0; i < n; ++i) {
        v[i] = .0f;
    }
}

void
vec_mul_add(const float* v1,
            const float alpha,
            const float beta,
            float* v2,
            size_t const n)
{
    for (size_t i = 0; i < n; ++i) {
        v2[i] = v1[i] * alpha + beta;
    }
}

bool
is_vec_eq(const float* v1, const float* v2, size_t const n)
{
    for (size_t i = 0; i < n; ++i) {
        if (fabs(v1[i] - v2[i]) > 1e-6) {
            fprintf(
              stderr, "(v1[%zu] = %f) != (v2[%zu] = %f)\n", i, v1[i], i, v2[i]);
            return false;
        }
    }

    return true;
}

constexpr int M = 1027;
constexpr int N = 1027;
constexpr int K = 519;
// write v1, v2
float val1[M * K] = {};
float val2[K * N] = {};
float val3[M * N] = {};

Eigen::Map<
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  A(val1, M, K);
Eigen::Map<
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  B(val2, K, N);
Eigen::Map<
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  C(val3, M, N);

int
main(const int argc, const char* argv[])
{
    random_vec(val1, M * K);
    random_vec(val2, K * M);
    random_vec(val3, M * N);

    GPUDevice gpu;
    if (gpu.init() != VK_SUCCESS) {
        fprintf(stderr, "init gpu failed.\n");
        return -1;
    }

    {
        VkTensor b1(1, M, K, &gpu, true);
        VkTensor b2(1, K, N, &gpu, true);
        VkTensor b3;

        if (b1.create() != VK_SUCCESS || b2.create() != VK_SUCCESS) {
            return -1;
        }

        Command command(&gpu);
        auto ret = command.init();
        if (ret != VK_SUCCESS) {
            fprintf(stderr, "failed at init command\n");
            return -1;
        }

        command.begin();
        command.upload(val1, M * K, b1);
        command.upload(val2, K * N, b2);

        MatMul matmul(&gpu, &command);
        if (matmul.init() != VK_SUCCESS) {
            fprintf(stderr, "failed at init op\n");
            return -1;
        }
        ret = matmul(b1, b2, b3);

        if (ret != VK_SUCCESS) {
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
