#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "core/pipeline.h"
#include "shaders/vkllama_comp_shaders.h"
#include <chrono>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>
#include <math.h>

void
random_vec(float* v, const int n)
{
    for (int i = 0; i < n; ++i) {
        v[i] = static_cast<float>(random() % 100) / 50.0f;
    }
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

TEST(matmul, test_matmul)
{
    random_vec(val1, M * K);
    random_vec(val2, K * M);
    random_vec(val3, M * N);

    GPUDevice gpu;
    ASSERT_EQ(gpu.init(), VK_SUCCESS) << "failed at init gpu";

    {
        VkTensor b1(1, M, K, &gpu, true);
        VkTensor b2(1, K, N, &gpu, true);
        VkTensor b3(1, M, N, &gpu, true);
        ASSERT_TRUE(b1.create() == VK_SUCCESS && b2.create() == VK_SUCCESS &&
                    b3.create() == VK_SUCCESS) << "failed at creating bindings";

        b3.set_access_flags(VK_ACCESS_SHADER_WRITE_BIT);
        b3.set_pipeline_stage(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        Pipeline::ShaderInfo shaderInfo = { 0, 3, 3, 16, 16, 1 };

        Pipeline::ConstantType m = { .i = M };
        Pipeline::ConstantType n = { .i = N };
        Pipeline::ConstantType k = { .i = K };

        Pipeline pipeline(&gpu,
                          __get_matmul_comp_spv_code(),
                          __get_matmul_comp_spv_size(),
                          {},
                          shaderInfo);

        ASSERT_TRUE(pipeline.init() == VK_SUCCESS) << "failed at init pipeline";

        ASSERT_TRUE(pipeline.set_group((N + 15) / 16, (M + 15) / 16, 1) ==
                    VK_SUCCESS) << "failed at setting work group";

        {
            CommandScope command(&gpu);
            command.upload(val1, M * K, b1);
            command.upload(val2, K * N, b2);
            command.upload(val3, M * N, b3);
            command.record_pipeline(pipeline, { b1, b2, b3 }, { m, n, k });
            command.download(b3, val3, M * N);
        }

        auto c = A * B;
        auto mse = (C - c).array().pow(2.f).mean();
        ASSERT_LT(mse, 1e-4) << "failed at testing result";
    }
}
