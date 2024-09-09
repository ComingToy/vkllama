#ifndef __VKLLAMA_MULTIHEADATTENTIONV2_H__
#define __VKLLAMA_MULTIHEADATTENTIONV2_H__
#include "src/core/tensor.h"
#include "src/ops/elementwise.h"
#include "src/ops/feed_forward.h"
#include "src/ops/mat_mul.h"
#include "src/ops/op.h"
#include "src/ops/read_kvcache_op.h"
#include "src/ops/rope.h"
#include "src/ops/slice.h"
#include "src/ops/softmax.h"
#include "src/ops/transpose.h"
#include "src/ops/update_kv_cache.h"
#include <memory>
#include <vector>

namespace vkllama
{
class GPUDevice;
class Command;

class MultiHeadAttentionV2 : public Op
{
public:
  MultiHeadAttentionV2 (GPUDevice *dev, Command *command, VkTensor wk,
                        VkTensor wq, VkTensor wv, VkTensor wo,
                        const int maxlen, const int dim,
                        const bool transposed_weight = false,
                        VkTensor::DType dtype = VkTensor::FP32,
                        const bool use_kvcache = false);

  absl::Status operator() (VkTensor X, VkTensor &out,
                           const size_t offset = 0) noexcept;
  absl::Status init () noexcept override;
  uint64_t time () noexcept override;

private:
  VkTensor wk_;
  VkTensor wq_;
  VkTensor wv_;
  VkTensor wo_;
  VkTensor kcache_;
  VkTensor vcache_;

  const int maxlen_;
  const int dim_;
  const bool transposed_weight_;
  VkTensor::DType dtype_;
  const bool use_kvcache_;

  std::unique_ptr<MatMul> matmul_k_;
  std::unique_ptr<MatMul> matmul_q_;
  std::unique_ptr<MatMul> matmul_v_;
  std::unique_ptr<MatMul> matmul_o_;
  std::unique_ptr<MatMul> matmul_qk_;
  std::unique_ptr<MatMul> matmul_weighted_;
  std::unique_ptr<ElementWise> scaled_;
  std::unique_ptr<Softmax> softmax_;
  std::unique_ptr<Rope> rope_;
  std::unique_ptr<Transpose> transpose_k_;
  std::unique_ptr<Transpose> transpose_q_;
  std::unique_ptr<Transpose> transpose_v_;
  std::unique_ptr<Transpose> transpose_heads_;
  std::unique_ptr<MatMul> matmul_attn_score_;
  std::unique_ptr<UpdateKVCache> update_kcache_op_;
  std::unique_ptr<UpdateKVCache> update_vcache_op_;
  std::unique_ptr<ReadKVCache> kcache_read_op_;
  std::unique_ptr<ReadKVCache> vcache_read_op_;

  std::vector<VkTensor> tmp_tensors_;
};
}

#endif
