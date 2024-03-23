#include "mat_mul.h"
#include "shaders/vkllama_comp_shaders.h"
#include "src/core/command.h"
#include "src/core/gpu_device.h"
#include "src/core/pipeline.h"
#include "src/core/tensor.h"

MatMul::MatMul(GPUDevice* dev, Command* command)
  : Op(dev, command)
{
    Pipeline::ShaderInfo info = { 0, 3, 3, 16, 16, 1 };

    pipeline_.reset(new Pipeline(dev_,
                                 __get_matmul_shared_mem_comp_spv_code(),
                                 __get_matmul_shared_mem_comp_spv_size(),
                                 {},
                                 info));
}

VkResult
MatMul::init()
{
    return pipeline_->init();
}

uint64_t
MatMul::time()
{
    return pipeline_->time();
}

VkResult
MatMul::operator()(VkTensor a, VkTensor b, VkTensor& c)
{
    c = VkTensor(
      std::max(a.channels(), b.channels()), a.height(), b.width(), dev_, false);

    auto ret = c.create();
    if (ret != VK_SUCCESS) {
        return ret;
    }

    c.set_access_flags(VK_ACCESS_SHADER_WRITE_BIT);
    c.set_pipeline_stage(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    Pipeline::ConstantType M = { .i = (int)a.height() };
    Pipeline::ConstantType N = { .i = (int)b.width() };
    Pipeline::ConstantType K = { .i = (int)b.height() };

    uint32_t groupx = (N.i + 31) / 32, groupy = (M.i + 31) / 32, groupz = 1;
    pipeline_->set_group(groupx, groupy, groupz);

    return command_->record_pipeline(*pipeline_, { a, b, c }, { M, N, K });
}
