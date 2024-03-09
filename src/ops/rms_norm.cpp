#include "rms_norm.h"
#include "shaders/vkllama_shaders.h"
#include "src/core/command.h"
#include "src/core/pipeline.h"

RMSNorm::RMSNorm(GPUDevice* dev, Command* command)
  : Op(dev, command)
{
    Pipeline::ConstantType power = { .f = 2.0f };
    Pipeline::ConstantType eps = { .f = 1e-3 };
    Pipeline::ShaderInfo info = { 2, 2, 3, 1, 32, 32 };
    pipeline_.reset(new Pipeline(dev_,
                                 __reduce_mean_comp_spv_code.pcode,
                                 __reduce_mean_comp_spv_code.size,
                                 { power, eps },
                                 info));
}

VkResult
RMSNorm::operator()(VkTensor a, VkTensor& b)
{
    b = VkTensor(a.channels(), a.height(), a.width(), dev_, false);
    auto ret = b.create();
    if (ret != VK_SUCCESS) {
        return ret;
    }

    b.set_access_flags(VK_ACCESS_SHADER_WRITE_BIT);
    b.set_pipeline_stage(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    Pipeline::ConstantType C = { .u32 = (uint32_t)a.channels() };
    Pipeline::ConstantType H = { .u32 = (uint32_t)a.height() };
    Pipeline::ConstantType W = { .u32 = (uint32_t)a.width() };

    pipeline_->set_group(1, (H.u32 + 31) / 32, (C.u32 + 31) / 32);
    return command_->record_pipeline(*pipeline_, { a, b }, { C, H, W });
}
