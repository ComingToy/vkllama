#ifndef __VKLLAMA_MATMUL_H__
#define __VKLLAMA_MATMUL_H__

#include "src/ops/op.h"
#include <memory>

class MatMul : public Op
{
public:
    MatMul(GPUDevice* dev, Command* command, const int act = 0, const int broadcast_type = 0);
    VkResult operator()(VkTensor a, VkTensor b, VkTensor& c);
    VkResult init() override;
    uint64_t time() override;

private:
    std::unique_ptr<Pipeline> pipeline_;
    const int broadcast_type_;
    const int act_;
};

#endif
