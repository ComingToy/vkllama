#ifndef __VKLLAMA_TEST_COMMON_H__
#define __VKLLAMA_TEST_COMMON_H__
#include <math.h>

#include <optional>
#include <tuple>
#include <vector>
#include <random>

#include "core/command.h"
#include "core/tensor.h"

inline void random_vec(float* v, const int n, const float min, const float max)
{
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<float> dist(min, max);
    for (int i = 0; i < n; ++i)
    {
        v[i] = dist(e2);
    }
}

inline std::optional<std::pair<VkTensor, std::vector<float> > > random_tensor(
    GPUDevice* dev, Command* command, const int c, const int h, const int w, const float min = -1.0, const float max = 1.0)
{
    VkTensor tensor(c, h, w, dev);
    if (tensor.create() != VK_SUCCESS)
    {
        return {};
    }

    const int n = c * h * w;
    std::vector<float> buf(n);
    random_vec(buf.data(), n, min, max);

    auto ret = command->upload(buf.data(), n, tensor);
    if (ret != VK_SUCCESS)
    {
        return {};
    }

    return {{tensor, buf}};
}
#endif
