#ifndef __VKLLAMA_ROPE_H__
#define __VKLLAMA_ROPE_H__

#include "src/ops/op.h"
#include <memory>
#include <vector>

class Rope : public Op
{
public:
  Rope (GPUDevice *dev, Command *command, const int maxlen, const int dim);
  VkResult operator() (VkTensor query, VkTensor key, VkTensor &out_query,
                       VkTensor &out_key) noexcept;
  VkResult init () noexcept override;
  uint64_t time () noexcept override;
  const std::vector<float> &freqc ();
  const std::vector<float> &freqs ();

private:
  const int maxlen_;
  const int dim_;
  void precompute_freq_ ();
  VkTensor freqc_;
  VkTensor freqs_;
  std::vector<float> freqc_host_;
  std::vector<float> freqs_host_;

  std::unique_ptr<Pipeline> pipeline_;
};

#endif
