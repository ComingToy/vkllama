#include "src/core/allocator.h"
#include "shaders/vkllama_comp_shaders.h"
#include "src/core/command.h"
#include "src/core/gpu_device.h"
#include "src/core/pipeline.h"
#include <cstdio>
#include "Eigen/Eigen"
#include <iostream>
#include <math.h>

void
random_vec(float* v, const int n)
{
    for (int i = 0; i < n; ++i) {
        v[i] = static_cast<float>(random() % 100);
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

constexpr size_t vec_size = 20000 / sizeof(float);

int
main(const int argc, const char* argv[])
{
    // write v1, v2
    float val1[vec_size] = {};
    float val2[vec_size] = {};
    random_vec(val1, vec_size);
    random_vec(val2, vec_size);

    Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      A(val1, 1, vec_size);

    Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      B(val2, 1, vec_size);

    GPUDevice gpu;
    if (gpu.init() != VK_SUCCESS) {
        fprintf(stderr, "failed at init gpu\n");
        return -1;
    }
    {
        VkTensor b1(1, 1, vec_size, &gpu, true);
        VkTensor b2(1, 1, vec_size, &gpu, true);

        if (b1.create() != VK_SUCCESS || b2.create() != VK_SUCCESS)
            return -1;

        b2.set_access_flags(VK_ACCESS_SHADER_WRITE_BIT);
        b2.set_pipeline_stage(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        Pipeline::ShaderInfo shaderInfo = { 0, 2, 3, 1024, 1, 1 };
        Pipeline::ConstantType alpha = { .f = 20.55f };
        Pipeline::ConstantType beta = { .f = 15.55f };
		Pipeline::ConstantType N = {.u32 = (uint32_t)vec_size};

        Pipeline pipeline(&gpu,
                          __get_vec_mul_add_comp_spv_code(),
                          __get_vec_mul_add_comp_spv_size(),
                          {},
                          shaderInfo);
        pipeline.init();

        if (pipeline.set_group((vec_size + 1023)/1024, 1, 1) != VK_SUCCESS) {
            fprintf(stderr, "set group failed.\n");
            return -1;
        }

        {
            CommandScope command(&gpu);
            command.upload(val1, vec_size, b1);
        }

        {
            Command command(&gpu);
           	command.init();
            for (int i = 0; i < 1000; ++i) {
                command.begin();
                command.record_pipeline(pipeline, { b1, b2 }, { alpha, beta, N });
                command.end();
                auto ret = command.submit_and_wait();
				if (ret != VK_SUCCESS){
					return -1;
				}
			}
        }

        {
            CommandScope command(&gpu);
            command.download(b2, val2, vec_size);
        }

        auto C = A.array() * alpha.f + beta.f;
        std::cerr << "time cost: " << pipeline.time() << std::endl;
        std::cerr << "mse: " << (C - B.array()).pow(2.f).mean() << std::endl;
    }
    return 0;
}
