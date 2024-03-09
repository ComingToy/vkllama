#include "src/core/command.h"
#include "src/core/gpu_device.h"
#include "src/ops/rms_norm.h"
#include <cstdio>
#include <Eigen/src/Core/Map.h>
#include <Eigen/src/Core/Matrix.h>

float x[3 * 1024 * 1024];
float output[3 * 1024 * 1024];

Eigen::Map<
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  A(x, 3 * 1024, 1024);

Eigen::Map<
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  B(output, 3 * 1024, 1024);

void
random_vec(float* v, const int n)
{
    for (int i = 0; i < n; ++i) {
        v[i] = 1.0;
    }
}

int
main(void)
{
    random_vec(x, 3 * 1024 * 1024);
    GPUDevice gpu;
    if (gpu.init() != VK_SUCCESS) {
        fprintf(stderr, "failed at init gpu device\n");
        return -1;
    }

    fprintf(stderr, "init gpu successfully!\n");

    {
        VkTensor a(3, 1024, 1024, &gpu, true);
        if (a.create() != VK_SUCCESS) {
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
		
        RMSNorm norm(&gpu, &command);
        ret = norm.init();

        if (ret != VK_SUCCESS) {
            fprintf(stderr, "failed at init op\n");
            return -1;
        }

        VkTensor b;
        ret = norm(a, b);
        if (ret != VK_SUCCESS) {
            fprintf(stderr, "failed at op compute\n");
            return -1;
        }

        command.download(b, output, 3 * 1024 * 1024);
        command.end();
        command.submit_and_wait();

        std::cerr << "time cost: " << norm.time() << std::endl;

        auto W = (A.array().pow(2.0).rowwise().mean() + 1e-3).rsqrt().rowwise().replicate(1024);

        std::cerr << "W.row " << W.rows() << ", W.cols " << W.cols()
                  << ", A.rows " << A.rows() << ", A.cols " << A.cols()
                  << std::endl;
        auto C = A.array() * W;

        auto mse = (C - B.array()).pow(2.0f).mean();
        std::cerr << "mse: " << mse << std::endl;
    }

    return 0;
}

