#include "rms_norm.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include "src/core/command.h"
#include "src/core/pipeline.h"

RMSNorm::RMSNorm(GPUDevice* dev, Command* command)
  : Op(dev, command)
{
    Pipeline::ConstantType power = { .f = 2.0f };
    Pipeline::ConstantType eps = { .f = 1e-3 };
    Pipeline::ShaderInfo info = { 2, 3, 3, 1, 32, 32 };
    pipeline_.reset(new Pipeline(dev_,
                                 __get_rms_norm_comp_spv_code(),
                                 __get_rms_norm_comp_spv_size(),
                                 { power, eps },
                                 info));
}

VkResult
RMSNorm::init()
{
    return pipeline_->init();
}

uint64_t
RMSNorm::time()
{
    return pipeline_->time();
}

VkResult
RMSNorm::operator()(VkTensor x, VkTensor w, VkTensor& output)
{
    output = VkTensor(x.channels(), x.height(), x.width(), dev_, false);
    auto ret = output.create();
    if (ret != VK_SUCCESS) {
        return ret;
    }

    output.set_access_flags(VK_ACCESS_SHADER_WRITE_BIT);
    output.set_pipeline_stage(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    Pipeline::ConstantType C = { .u32 = (uint32_t)x.channels() };
    Pipeline::ConstantType H = { .u32 = (uint32_t)x.height() };
    Pipeline::ConstantType W = { .u32 = (uint32_t)x.width() };

    pipeline_->set_group(1, (H.u32 + 31) / 32, (C.u32 + 31) / 32);
    return command_->record_pipeline(*pipeline_, { x, w, output }, { C, H, W });
}
