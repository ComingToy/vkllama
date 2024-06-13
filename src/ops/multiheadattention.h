#ifndef __VKLLAMA_MULTIHEADATTENTION_H__
#define __VKLLAMA_MULTIHEADATTENTION_H__
#include "src/core/tensor.h"
#include "src/ops/concat.h"
#include "src/ops/elementwise.h"
#include "src/ops/mat_mul.h"
#include "src/ops/op.h"
#include "src/ops/rope.h"
#include "src/ops/slice.h"
#include "src/ops/softmax.h"
#include "src/ops/update_kv_cache.h"
#include <memory>
#include <vector>

namespace vkllama
{
class GPUDevice;
class Command;
class MultiHeadAttention : public Op
{
public:
  MultiHeadAttention (GPUDevice *dev, Command *command,
                      std::vector<VkTensor> const &Wk,
                      std::vector<VkTensor> const &Wq,
                      std::vector<VkTensor> const &Wv, VkTensor const Wo,
                      const int maxlen, const int dim,
                      const bool transposed_weight = false,
                      VkTensor::DType const dtype = VkTensor::FP32,
                      const bool use_kvcache = false);
  VkResult operator() (VkTensor X, VkTensor &output,
                       const size_t offset = 0) noexcept;
  VkResult init () noexcept override;
  uint64_t time () noexcept override;

private:
  std::vector<std::unique_ptr<MatMul> > k_ops_;
  std::vector<std::unique_ptr<MatMul> > q_ops_;
  std::vector<std::unique_ptr<MatMul> > v_ops_;
  std::vector<std::unique_ptr<MatMul> > weighted_ops_;
  std::vector<std::unique_ptr<MatMul> > attn_ops_;
  std::vector<std::unique_ptr<Rope> > rope_ops_;
  std::vector<std::unique_ptr<ElementWise> > elementwise_ops_;
  std::vector<std::unique_ptr<Softmax> > softmax_ops_;
  std::unique_ptr<MatMul> out_matmul_;
  std::unique_ptr<Concat> concat_;
  std::vector<std::unique_ptr<UpdateKVCache> > update_cache_ops_;
  std::vector<std::unique_ptr<Slice> > cache_slice_ops_;

  std::vector<VkTensor> wk_;
  std::vector<VkTensor> wq_;
  std::vector<VkTensor> wv_;
  VkTensor wo_;

  std::vector<VkTensor> tmp_tensors_;
  int maxlen_;
  int dim_;
  bool transposed_weight_;

  const VkTensor::DType dtype_;
  const bool use_kvcache_;

  VkTensor kcache_;
  VkTensor vcache_;
};
}

#endif
