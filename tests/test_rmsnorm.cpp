#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/rms_norm.h"
#include <cstdio>

float x[3 * 1024 * 1024];
float w[1024];
float output[3 * 1024 * 1024];

Eigen::Map<
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  A(x, 3 * 1024, 1024);

Eigen::Map<
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  W(w, 1, 1024);

Eigen::Map<
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  B(output, 3 * 1024, 1024);

void
random_vec(float* v, const int n)
{
    for (int i = 0; i < n; ++i) {
        v[i] = static_cast<float>(random() % 100) / 50.0f;
    }
}

int
main(void)
{
    random_vec(x, 3 * 1024 * 1024);
    random_vec(w, 1024);
    GPUDevice gpu;
    if (gpu.init() != VK_SUCCESS) {
        fprintf(stderr, "failed at init gpu device\n");
        return -1;
    }

    fprintf(stderr, "init gpu successfully!\n");

    {
        VkTensor a(3, 1024, 1024, &gpu, true);
        VkTensor b(1, 1, 1024, &gpu, true);
        if (a.create() != VK_SUCCESS || b.create() != VK_SUCCESS) {
            fprintf(stderr, "failed at create input tensor\n");
            return -1;
        }

        Command command(&gpu);
        auto ret = command.init();

        if (ret != VK_SUCCESS) {
            fprintf(stderr, "failed at init command\n");
            return -1;
        }

        command.begin();
        command.upload(x, 3 * 1024 * 1024, a);
        command.upload(w, 1024, b);

        RMSNorm norm(&gpu, &command);
        ret = norm.init();

        if (ret != VK_SUCCESS) {
            fprintf(stderr, "failed at init op\n");
            return -1;
        }

        VkTensor c;
        ret = norm(a, b, c);
        if (ret != VK_SUCCESS) {
            fprintf(stderr, "failed at op compute\n");
            return -1;
        }

        command.download(c, output, 3 * 1024 * 1024);
        command.end();
        command.submit_and_wait();

        std::cerr << "time cost: " << norm.time() << std::endl;

        auto V = (A.array().pow(2.0).rowwise().mean() + 1e-3)
                   .rsqrt()
                   .rowwise()
                   .replicate(1024);

        auto C = A.array() * V * W.array().replicate<3 * 1024, 1>();

        auto mse = (C - B.array()).pow(2.0f).mean();
        std::cerr << "mse: " << mse << std::endl;
    }

    return 0;
}

