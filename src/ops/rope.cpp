#include "src/ops//rope.h"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>
#include <vector>
#include "src/core/command.h"
#include "src/core/pipeline.h"
#include "shaders/vkllama_comp_shaders.h"

Rope::Rope(GPUDevice* dev, Command* command, const int maxlen, const int dim)
    : Op(dev, command), maxlen_(maxlen), dim_(dim)
{
    freqc_host_.resize(maxlen_ * dim_/2);
    freqs_host_.resize(maxlen_ * dim_/2);
}

void Rope::precompute_freq_()
{
    std::vector<float> freq;
    std::generate_n(std::back_inserter(freq), dim_ / 2, [this, n = 0]() mutable {
        float dim = dim_;
        float f = 1.0f / std::pow(10000.0f, static_cast<float>(n) / dim);
        n += 2;
        return f;
    });

    // [seqlen, headim]
    for (int i = 0; i < maxlen_; ++i)
    {
        for (int k = 0; k < dim_ / 2; ++k)
        {
            auto f = freq[k] * static_cast<float>(i);
            freqc_host_[i * dim_ / 2 + k] = std::cos(f);
            freqs_host_[i * dim_ / 2 + k] = std::sin(f);
        }
    }
}

VkResult Rope::init()
{
    precompute_freq_();
    freqc_ = VkTensor(1, maxlen_, dim_, dev_);
    freqs_ = VkTensor(1, maxlen_, dim_, dev_);

    VkResult ret = VK_SUCCESS;
    if ((ret = freqc_.create()) != VK_SUCCESS || (ret = freqs_.create()) != VK_SUCCESS)
    {
        return ret;
    }

    ret = command_->upload(freqc_host_.data(), freqc_host_.size(), freqc_);
    if (ret != VK_SUCCESS) { return ret; }
    ret = command_->upload(freqs_host_.data(), freqs_host_.size(), freqs_);
    if (ret != VK_SUCCESS)
    {
        return ret;
    }

    Pipeline::ShaderInfo shaderInfo = {0, 6, 3, 16, 16, 1};
    pipeline_.reset(new Pipeline(dev_, __get_rope_comp_spv_code(), __get_rope_comp_spv_size(), {}, shaderInfo));

    return pipeline_->init();
}

uint64_t Rope::time()
{
    return pipeline_->time();
}

VkResult Rope::operator()(VkTensor query, VkTensor key, VkTensor& out_query, VkTensor& out_key)
{
    if (query.width() != key.width() || query.height() != key.height() || query.channels() != key.channels() || query.width() != dim_ || query.height() > maxlen_)
    {
        return VK_ERROR_UNKNOWN;
    }

    out_query = VkTensor::like(query);
    out_key = VkTensor::like(key);
    auto ret = out_query.create();
    if (ret != VK_SUCCESS) { return ret; }
    if ((ret = out_key.create()) != VK_SUCCESS) { return ret; }

    uint32_t groupx = (query.width() / 2 + 15) / 16 * 16, groupy = (query.height() + 15) / 16 * 16, groupz = query.channels();
    ret = pipeline_->set_group(groupx, groupy, groupz);
    if (ret != VK_SUCCESS) { return ret; }

    Pipeline::ConstantType C = {.u32 = (uint32_t)query.channels()};
    Pipeline::ConstantType H = {.u32 = (uint32_t)query.height()};
    Pipeline::ConstantType W = {.u32 = (uint32_t)query.width()};

    ret = command_->record_pipeline(*pipeline_, {query, key, freqc_, freqs_, out_query, out_key}, {C, H, W});
    if (ret != VK_SUCCESS)
    {
        return ret;
    }

    out_query.set_access_flags(VK_ACCESS_SHADER_WRITE_BIT);
    out_query.set_pipeline_stage(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    out_key.set_access_flags(VK_ACCESS_SHADER_WRITE_BIT);
    out_key.set_pipeline_stage(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    return VK_SUCCESS;
}

const std::vector<float>& Rope::freqc()
{
    return freqc_host_;
}

const std::vector<float>& Rope::freqs()
{
    return freqs_host_;
}
