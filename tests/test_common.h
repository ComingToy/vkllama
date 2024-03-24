#ifndef __VKLLAMA_TEST_COMMON_H__
#define __VKLLAMA_TEST_COMMON_H__
#include "core/command.h"
#include "core/tensor.h"
#include <math.h>
#include <optional>
#include <vector>

inline void
random_vec(float* v, const int n)
{
    for (int i = 0; i < n; ++i) {
        v[i] = static_cast<float>(random() % 100) / 50.0f;
    }
}

inline std::optional<VkTensor>
random_tensor(GPUDevice* dev,
              Command* command,
              const int c,
              const int h,
              const int w)
{
    VkTensor tensor(c, h, w, dev);
    if (tensor.create() != VK_SUCCESS) {
        return {};
    }

    const int n = c * h * w;
    std::vector<float> buf(n);
    random_vec(buf.data(), n);

    auto ret = command->upload(buf.data(), n, tensor);
    if (ret != VK_SUCCESS) {
        return {};
    }

    return tensor;
}
#endif
