#include "shaders/vkllama_comp_shaders.h"
#include "src/core/allocator.h"
#include "src/core/command.h"
#include "src/core/gpu_device.h"
#include "src/core/pipeline.h"
#include <chrono>
#include <cstdio>
#include "Eigen/Eigen"
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

int
main(const int argc, const char* argv[])
{
    random_vec(val1, M * K);
    random_vec(val2, K * M);
    random_vec(val3, M * N);

    GPUDevice gpu;
	if (gpu.init() != VK_SUCCESS)
	{
		fprintf(stderr, "failed at init gpu\n");
		return -1;
	}

    {
        VkTensor b1(1, M, K, &gpu, true);
        VkTensor b2(1, K, N, &gpu, true);
        VkTensor b3(1, M, N, &gpu, true);
        if (b1.create() != VK_SUCCESS || b2.create() != VK_SUCCESS ||
            b3.create() != VK_SUCCESS) {
            return -1;
        }

        b3.set_access_flags(VK_ACCESS_SHADER_WRITE_BIT);
        b3.set_pipeline_stage(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        Pipeline::ShaderInfo shaderInfo = { 0, 3, 3, 16, 16, 1 };

        Pipeline::ConstantType m = { .i = M };
        Pipeline::ConstantType n = { .i = N };
        Pipeline::ConstantType k = { .i = K };

        Pipeline pipeline(&gpu,
                          get_matmul_comp_spv_code(),
                          get_matmul_comp_spv_size(),
                          {},
                          shaderInfo);

        if (pipeline.init() != VK_SUCCESS) {
            fprintf(stderr, "init pipeline failed.\n");
            return -1;
        }

        if (pipeline.set_group((N + 15) / 16, (M + 15) / 16, 1) != VK_SUCCESS) {
            fprintf(stderr, "set group failed.\n");
            return -1;
        }

        fprintf(stderr,
                "pipeline group_x = %d, gorup_y = %d, group_z = %d\n",
                pipeline.group_x(),
                pipeline.group_y(),
                pipeline.group_z());

        {
            CommandScope command(&gpu);
            command.upload(val1, M * K, b1);
            command.upload(val2, K * N, b2);
            command.upload(val3, M * N, b3);
            command.record_pipeline(pipeline, { b1, b2, b3 }, { m, n, k });
            command.download(b3, val3, M * N);
        }

        std::cerr << "time cost: " << pipeline.time() << std::endl;
        auto c = A * B;
        auto mse = (C - c).array().pow(2.f).mean();
        std::cerr << "mse: " << mse << std::endl;
    }
    return 0;
}
