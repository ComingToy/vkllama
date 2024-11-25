#ifndef __VKLLAMA_MULTIHEADATTENTIONV2_H__
#define __VKLLAMA_MULTIHEADATTENTIONV2_H__
#include "src/core/tensor.h"
#include "src/ops/elementwise.h"
#include "src/ops/feed_forward.h"
#include "src/ops/mat_mul.h"
#include "src/ops/op.h"
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
  MultiHeadAttentionV2 (GPUDevice *dev, Command *command, Tensor wk, Tensor wq,
                        Tensor wv, Tensor wo, const int maxlen, const int dim,
                        const bool transposed_weight = false,
                        Tensor::DType dtype = FP16,
                        const bool use_kvcache = false,
                        const bool clip_output = false);

  absl::StatusOr<Tensor> operator() (Tensor X,
                                     const size_t offset = 0) noexcept;
  absl::Status init () noexcept override;
  uint64_t time () noexcept override;

private:
  Tensor wk_;
  Tensor wq_;
  Tensor wv_;
  Tensor wo_;
  Tensor kcache_;
  Tensor vcache_;

  const int maxlen_;
  const int dim_;
  const bool transposed_weight_;
  Tensor::DType dtype_;
  const bool use_kvcache_;
  const bool clip_output_;

  std::unique_ptr<MatMul> matmul_o_;
  std::unique_ptr<MatMul> matmul_qk_;
  std::unique_ptr<MatMul> matmul_weighted_;
  std::unique_ptr<ElementWise> scaled_;
  std::unique_ptr<Softmax> softmax_;
  std::unique_ptr<Rope> rope_q_;
  std::unique_ptr<Rope> rope_k_;
  std::unique_ptr<Transpose> transpose_k_;
  std::unique_ptr<Transpose> transpose_q_;
  std::unique_ptr<Transpose> transpose_v_;
  std::unique_ptr<Transpose> transpose_heads_;
  std::unique_ptr<MatMul> matmul_attn_score_;
  std::unique_ptr<UpdateKVCache> update_kcache_op_;
  std::unique_ptr<UpdateKVCache> update_vcache_op_;
  std::unique_ptr<Slice> clip_output_op_;
  std::unique_ptr<Pipeline> kqv_pipeline_;

  std::vector<Tensor> tmp_tensors_;
};
}

#endif
