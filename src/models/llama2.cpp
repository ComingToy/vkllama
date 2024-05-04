#include "src/core/command.h"
#include "src/core/tensor.h"
#include "src/ops/embedding.h"
#include "src/ops/feed_forward.h"
#include "src/ops/multiheadattention.h"
#include "src/ops/rms_norm.h"
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

struct Variable
{
  void *data;
  const int c;
  const int h;
  const int w;
  VkTensor::DType dtype;
};

class InputLayer
{
public:
  InputLayer (GPUDevice *gpu, Command *command, VkTensor vocab,
              VkTensor weight, uint32_t UNK = 0)
      : vocab_ (vocab), norm_weights_ (weight), UNK_ (UNK)
  {
  }

  VkResult
  init ()
  {
    embedding_op_.reset (new Embedding (gpu_, command_, UNK_));
    auto ret = embedding_op_->init ();
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    norm_op_.reset (new RMSNorm (gpu_, command_));
    if ((ret = norm_op_->init ()) != VK_SUCCESS)
      {
        return ret;
      }

    return VK_SUCCESS;
  }

  VkTensor
  operator() (VkTensor toks)
  {
    VkTensor out;
    auto ret = embedding_op_->operator() (vocab_, toks, embs_);
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at forwarding embedding op");
      }

    if ((ret = norm_op_->operator() (embs_, norm_weights_, out)) != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at forwarding rms norm op");
      }

    return out;
  }

private:
  GPUDevice *gpu_;
  Command *command_;
  VkTensor vocab_;
  VkTensor norm_weights_;
  VkTensor embs_;

  uint32_t UNK_;
  std::unique_ptr<Embedding> embedding_op_;
  std::unique_ptr<RMSNorm> norm_op_;
};

class Llama2Block
{
public:
  struct TransformerParams
  {

    std::vector<VkTensor> Wk;
    std::vector<VkTensor> Wq;
    std::vector<VkTensor> Wv;
    VkTensor Wo;
    int maxlen;
    int dim;
  };

  struct FeedForwardParams
  {
    VkTensor w1;
    VkTensor w2;
    VkTensor w3;
  };

  struct RmsNormParams
  {
    VkTensor weight;
  };

  Llama2Block (GPUDevice *dev, Command *command,
               TransformerParams const &transformer,
               FeedForwardParams const &feed_forward,
               RmsNormParams const &norm)
      : gpu_ (dev), command_ (command), transformer_params_ (transformer),
        feedforward_params_ (feed_forward), rmsnorm_params_ (norm)
  {
  }

  VkResult
  init ()
  {
    attn_op_.reset (new MultiHeadAttention (
        gpu_, command_, transformer_params_.Wk, transformer_params_.Wq,
        transformer_params_.Wv, transformer_params_.Wo,
        transformer_params_.maxlen, transformer_params_.dim));
    feedforward_op_.reset (
        new FeedForward (gpu_, command_, feedforward_params_.w1,
                         feedforward_params_.w2, feedforward_params_.w3));
    norm_op_.reset (new RMSNorm (gpu_, command_));

    auto ret = attn_op_->init ();
    if (ret != VK_SUCCESS || (ret = feedforward_op_->init ()) != VK_SUCCESS
        || (ret = norm_op_->init ()) != VK_SUCCESS)
      {
        return ret;
      }

    return VK_SUCCESS;
  }

  VkTensor
  operator() (VkTensor in)
  {
    auto ret = norm_op_->operator() (in, rmsnorm_params_.weight, normed_);
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at forwarding RMSNorm op");
      }

    ret = attn_op_->operator() (normed_, transformed_);
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error (
            "failed at forwarding MultiHeadAttention op");
      }

    VkTensor out;
    ret = feedforward_op_->operator() (transformed_, out);
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at forwarding FeedForward op");
      }

    return out;
  }

private:
  GPUDevice *gpu_;
  Command *command_;
  std::unique_ptr<MultiHeadAttention> attn_op_;
  std::unique_ptr<FeedForward> feedforward_op_;
  std::unique_ptr<RMSNorm> norm_op_;

  TransformerParams transformer_params_;
  FeedForwardParams feedforward_params_;
  RmsNormParams rmsnorm_params_;

  VkTensor normed_;
  VkTensor transformed_;
};

class OutputLayer
{
public:
  OutputLayer (GPUDevice *gpu, Command *command, VkTensor wo)
      : gpu_ (gpu), command_ (command), wo_ (wo)
  {
  }

  VkResult
  init ()
  {
    matmul_op_.reset (new MatMul (gpu_, command_, 0, 0, true));
    return matmul_op_->init ();
  }

  VkTensor
  operator() (VkTensor in)
  {
    VkTensor out;
    auto ret = matmul_op_->operator() (in, wo_, out);
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at forwarding MatMul op");
      }

    return out;
  }

private:
  GPUDevice *gpu_;
  Command *command_;
  VkTensor wo_;
  std::unique_ptr<MatMul> matmul_op_;
};

class Model
{
public:
  Model ()
  {
    gpu_ = new GPUDevice ();
    command_ = new Command (gpu_);
  }

  VkResult
  init (std::map<std::string, Variable> const &variables)
  {
    auto ret = gpu_->init ();
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    if ((ret = command_->init ()) != VK_SUCCESS)
      {
        return ret;
      }

    return VK_SUCCESS;
  }

private:
  GPUDevice *gpu_;
  Command *command_;
};
