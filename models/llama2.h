#ifndef __VKLLAMA_MODELS_LLAMA2_H__
#define __VKLLAMA_MODELS_LLAMA2_H__
// clang-format off
#include <stddef.h>
extern "C"{
#include "gguflib.h"
}
// clang-format on
#include "src/core/command.h"
#include "src/core/float.h"
#include "src/core/tensor.h"
#include "src/ops/argop.h"
#include "src/ops/cast.h"
#include "src/ops/elementwise.h"
#include "src/ops/embedding.h"
#include "src/ops/feed_forward.h"
#include "src/ops/multiheadattention.h"
#include "src/ops/multiheadattention_v2.h"
#include "src/ops/rms_norm.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#define __VKLLAMA_LOG_COST 1
namespace vkllama
{
class InputLayer
{
public:
  InputLayer (GPUDevice *gpu, Command *command, VkTensor vocab,
              uint32_t UNK = 0)
      : gpu_ (gpu), command_ (command), vocab_ (vocab), UNK_ (UNK)
  {
  }

  VkResult
  init ()
  {
    embedding_op_.reset (
        new Embedding (gpu_, command_, vocab_, UNK_, VkTensor::FP16));
    auto ret = embedding_op_->init ();
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    return VK_SUCCESS;
  }

  VkTensor
  operator() (VkTensor toks)
  {
    VkTensor out;
    auto ret = embedding_op_->operator() (toks, out);
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at forwarding embedding op");
      }

    return out;
  }

private:
  GPUDevice *gpu_;
  Command *command_;
  VkTensor vocab_;
  VkTensor embs_;

  uint32_t UNK_;
  std::unique_ptr<Embedding> embedding_op_;
};

class Llama2Block
{
public:
  struct TransformerParams
  {
    VkTensor Wk;
    VkTensor Wq;
    VkTensor Wv;
    VkTensor Wo;
    int maxlen;
    int dim;
  };

  struct FeedForwardParams
  {
    VkTensor w1;
    VkTensor w2;
    VkTensor w3;
    float eps;
  };

  struct RmsNormParams
  {
    VkTensor weight1;
    VkTensor weight2;
    float eps;
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
    attn_op_.reset (new MultiHeadAttentionV2 (
        gpu_, command_, transformer_params_.Wk, transformer_params_.Wq,
        transformer_params_.Wv, transformer_params_.Wo,
        transformer_params_.maxlen, transformer_params_.dim, true,
        VkTensor::FP16, true));

    feedforward_op_.reset (new FeedForward (
        gpu_, command_, feedforward_params_.w1, feedforward_params_.w2,
        feedforward_params_.w3, true, VkTensor::FP16));

    norm_op_.reset (new RMSNorm (gpu_, command_, rmsnorm_params_.weight1,
                                 rmsnorm_params_.eps, VkTensor::FP16));
    norm_op2_.reset (new RMSNorm (gpu_, command_, rmsnorm_params_.weight2,
                                  feedforward_params_.eps, VkTensor::FP16));
    add_op_.reset (new ElementWise (gpu_, command_, 0, VkTensor::FP16));
    add_op2_.reset (new ElementWise (gpu_, command_, 0, VkTensor::FP16));

    auto ret = attn_op_->init ();
    if (ret != VK_SUCCESS || (ret = feedforward_op_->init ()) != VK_SUCCESS
        || (ret = norm_op_->init ()) != VK_SUCCESS
        || (ret = norm_op2_->init ()) != VK_SUCCESS
        || (ret = add_op_->init ()) != VK_SUCCESS
        || (ret = add_op2_->init ()) != VK_SUCCESS)
      {
        return ret;
      }

    return VK_SUCCESS;
  }

  VkTensor
  operator() (VkTensor in, const size_t offset)
  {
    auto ret = norm_op_->operator() (in, normed_);
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at forwarding RMSNorm op");
      }

    ret = attn_op_->operator() (normed_, transformed_, offset);
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error (
            "failed at forwarding MultiHeadAttention op");
      }

    ret = add_op_->operator() (transformed_, in, added_);
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at forwarding add op");
      }

    ret = norm_op2_->operator() (added_, normed2_);
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at forwarding RMSNorm op");
      }

    VkTensor out;
    ret = feedforward_op_->operator() (normed2_, feed_);
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at forwarding FeedForward op");
      }

    ret = add_op2_->operator() (feed_, added_, out);
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at forwarding add op");
      }

    return out;
  }

  void
  print_op_cost ()
  {
    fprintf (stderr,
             "attn norm cost: %llu, attn cost: %lld, attn add cost: %lld, "
             "fffn norm cost: %lld, ffn cost: %lld, ffn add cost: %lld\n",
             norm_op_->time (), attn_op_->time (), add_op_->time (),
             norm_op2_->time (), feedforward_op_->time (), add_op2_->time ());
  }

private:
  GPUDevice *gpu_;
  Command *command_;
  std::unique_ptr<MultiHeadAttentionV2> attn_op_;
  std::unique_ptr<FeedForward> feedforward_op_;
  std::unique_ptr<RMSNorm> norm_op_;
  std::unique_ptr<RMSNorm> norm_op2_;
  std::unique_ptr<ElementWise> add_op_;
  std::unique_ptr<ElementWise> add_op2_;

  TransformerParams transformer_params_;
  FeedForwardParams feedforward_params_;
  RmsNormParams rmsnorm_params_;

  VkTensor normed_;
  VkTensor normed2_;
  VkTensor transformed_;
  VkTensor added_;
  VkTensor feed_;
};

class OutputLayer
{
public:
  OutputLayer (GPUDevice *gpu, Command *command, VkTensor wo,
               VkTensor norm_weight)
      : gpu_ (gpu), command_ (command), wo_ (wo), norm_weight_ (norm_weight)
  {
  }

  VkResult
  init ()
  {
    matmul_op_.reset (
        new MatMul (gpu_, command_, wo_, 1.0, .0, 0, 0, true, VkTensor::FP16));
    auto ret = matmul_op_->init ();
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    norm_op_.reset (
        new RMSNorm (gpu_, command_, norm_weight_, 1e-6, VkTensor::FP16));
    if ((ret = norm_op_->init ()) != VK_SUCCESS)
      {
        return ret;
      }

    argmax_op_.reset (new ArgMax (gpu_, command_, VkTensor::FP16));
    return argmax_op_->init ();
  }

  VkTensor
  operator() (VkTensor in)
  {
    auto ret = norm_op_->operator() (in, norm_output_);
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at forwarding rms norm");
      }

    ret = matmul_op_->operator() (norm_output_, matmul_output_);
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at forwarding MatMul op");
      }

    VkTensor out;
    if (argmax_op_->operator() (matmul_output_, out) != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at forwarding argmax");
      }

    return out;
  }

private:
  GPUDevice *gpu_;
  Command *command_;
  VkTensor wo_;
  VkTensor norm_weight_;
  VkTensor norm_output_;
  VkTensor matmul_output_;
  std::unique_ptr<MatMul> matmul_op_;
  std::unique_ptr<RMSNorm> norm_op_;
  std::unique_ptr<ArgMax> argmax_op_;
};

class Model
{
public:
  Model ()
      : gpu_ (nullptr), input_command_ (nullptr), output_command_ (nullptr)
  {
  }

  ~Model ()
  {
    delete input_layer_;
    delete output_layer_;
    for (auto *block : blocks_)
      {
        delete block;
      }

    delete input_command_;
    delete output_command_;
    for (auto c : block_commands_)
      {
        delete c;
      }
    delete gpu_;
  }

  VkResult
  init (std::map<std::string, gguf_value *> &kv,
        std::map<std::string, gguf_tensor> &tensors)
  {
    gpu_ = new GPUDevice ();
    auto ret = gpu_->init ();
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    input_command_ = new Command (gpu_);
    output_command_ = new Command (gpu_);
    if ((ret = input_command_->init ()) != VK_SUCCESS
        || (ret = output_command_->init ()) != VK_SUCCESS)
      {
        return ret;
      }

    auto head_count = kv["llama.attention.head_count"]->uint32;
    auto block_count = kv["llama.block_count"]->uint32;
    auto norm_eps = kv["llama.attention.layer_norm_rms_epsilon"]->float32;
    auto maxlen = kv["llama.context_length"]->uint32;

    input_command_->begin ();
    output_command_->begin ();

    // input layer
    {
      auto embeddings = tensors["token_embd.weight"];
      auto output_weight = tensors["output.weight"];
      auto norm_weight = tensors["output_norm.weight"];

      VkTensor vkembeddings (1, embeddings.dim[1], embeddings.dim[0], gpu_,
                             VkTensor::FP16);
      VkTensor vkoutput_weight (1, output_weight.dim[1], output_weight.dim[0],
                                gpu_, VkTensor::FP16);
      VkTensor vknorm_weight (1, 1, norm_weight.dim[0], gpu_, VkTensor::FP16);

      std::vector<__vkllama_fp16_t> norm_weight_fp16;
      const float *p
          = reinterpret_cast<const float *> (norm_weight.weights_data);
      for (size_t i = 0; i < norm_weight.num_weights; ++i)
        {
          __vkllama_fp16_t v = { .u16 = __fp32_to_fp16 (p[i]) };
          norm_weight_fp16.push_back (v);
        }

      VkResult ret = VK_SUCCESS;
      if ((ret = vkembeddings.create ()) != VK_SUCCESS
          || (ret = vkoutput_weight.create ()) != VK_SUCCESS
          || (ret = vknorm_weight.create ()) != VK_SUCCESS)
        {
          return ret;
        }

      ret = input_command_->upload (
          (__vkllama_fp16_t *)embeddings.weights_data, embeddings.num_weights,
          vkembeddings);
      if (ret != VK_SUCCESS)
        {
          return ret;
        }

      ret = output_command_->upload (
          (__vkllama_fp16_t *)output_weight.weights_data,
          output_weight.num_weights, vkoutput_weight);
      if (ret != VK_SUCCESS)
        {
          return ret;
        }

      ret = input_command_->upload (norm_weight_fp16.data (),
                                    norm_weight.num_weights, vknorm_weight);
      if (ret != VK_SUCCESS)
        {
          return ret;
        }

      {
        uint32_t UNK = 0;
        input_layer_
            = new InputLayer (gpu_, input_command_, vkembeddings, UNK);
      }

      output_layer_ = new OutputLayer (gpu_, output_command_, vkoutput_weight,
                                       vknorm_weight);

      if ((ret = input_layer_->init ()) != VK_SUCCESS
          || (ret = output_layer_->init ()) != VK_SUCCESS)
        {
          return ret;
        }
      input_command_->end ();
      output_command_->end ();
      input_command_->submit ();
      output_command_->submit ();
    }

    // blocks
    {

      char vname[512];
      for (uint32_t b = 0; b < block_count; ++b)
        {

          ::snprintf (vname, sizeof (vname), "blk.%u.attn_norm.weight", b);
          const auto attn_norm_weight = tensors[vname];

          ::snprintf (vname, sizeof (vname), "blk.%u.attn_k.weight", b);
          const auto attn_k_weight = tensors[vname];

          ::snprintf (vname, sizeof (vname), "blk.%u.attn_q.weight", b);
          const auto attn_q_weight = tensors[vname];

          ::snprintf (vname, sizeof (vname), "blk.%u.attn_v.weight", b);
          const auto attn_v_weight = tensors[vname];

          ::snprintf (vname, sizeof (vname), "blk.%u.attn_output.weight", b);
          const auto attn_output_weight = tensors[vname];

          ::snprintf (vname, sizeof (vname), "blk.%u.ffn_norm.weight", b);
          const auto ffn_norm_weight = tensors[vname];

          ::snprintf (vname, sizeof (vname), "blk.%u.ffn_up.weight", b);
          const auto ffn_up_weight = tensors[vname];

          ::snprintf (vname, sizeof (vname), "blk.%u.ffn_down.weight", b);
          const auto ffn_down_weight = tensors[vname];

          ::snprintf (vname, sizeof (vname), "blk.%u.ffn_gate.weight", b);

          const auto ffn_gate_weight = tensors[vname];

          auto command = new Command (gpu_);
          if ((ret = command->init ()) != VK_SUCCESS
              || (ret = command->begin ()) != VK_SUCCESS)
            {
              return ret;
            }
          block_commands_.push_back (command);

          VkTensor vk_attn_norm_weight (1, 1, attn_norm_weight.dim[0], gpu_,
                                        VkTensor::FP16);

          VkTensor vk_ffn_norm_weight (1, 1, ffn_norm_weight.dim[0], gpu_,
                                       VkTensor::FP16);

          VkResult ret = VK_SUCCESS;
          if ((ret = vk_attn_norm_weight.create ()) != VK_SUCCESS
              || (ret = vk_ffn_norm_weight.create ()) != VK_SUCCESS)
            {
              return ret;
            }

          std::vector<__vkllama_fp16_t> attn_norm_weight_fp16;
          const float *p = reinterpret_cast<const float *> (
              attn_norm_weight.weights_data);
          for (size_t i = 0; i < attn_norm_weight.num_weights; ++i)
            {
              attn_norm_weight_fp16.push_back (
                  { .u16 = __fp32_to_fp16 (p[i]) });
            }

          ret = command->upload (attn_norm_weight_fp16.data (),
                                 attn_norm_weight_fp16.size (),
                                 vk_attn_norm_weight);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          std::vector<__vkllama_fp16_t> ffn_norm_weight_fp16;
          p = reinterpret_cast<const float *> (ffn_norm_weight.weights_data);
          for (size_t i = 0; i < ffn_norm_weight.num_weights; ++i)
            {
              ffn_norm_weight_fp16.push_back (
                  { .u16 = __fp32_to_fp16 (p[i]) });
            }

          ret = command->upload (ffn_norm_weight_fp16.data (),
                                 ffn_norm_weight_fp16.size (),
                                 vk_ffn_norm_weight);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          size_t head_dim = attn_k_weight.dim[1];
          size_t input_dim = attn_k_weight.dim[0];
          size_t head_weight_size = head_dim * input_dim;

          VkTensor vkWk (1, head_dim, input_dim, gpu_, VkTensor::FP16);
          VkTensor vkWq (1, head_dim, input_dim, gpu_, VkTensor::FP16);
          VkTensor vkWv (1, head_dim, input_dim, gpu_, VkTensor::FP16);

          if ((ret = vkWk.create ()) != VK_SUCCESS
              || (ret = vkWq.create ()) != VK_SUCCESS
              || (ret = vkWv.create ()) != VK_SUCCESS)
            {
              return ret;
            }

          const __vkllama_fp16_t *wk_weight_data
              = (__vkllama_fp16_t *)attn_k_weight.weights_data;

          const __vkllama_fp16_t *wq_weight_data
              = (__vkllama_fp16_t *)attn_q_weight.weights_data;

          const __vkllama_fp16_t *wv_weight_data
              = (__vkllama_fp16_t *)attn_v_weight.weights_data;

          ret = command->upload (wk_weight_data, head_weight_size, vkWk);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          ret = command->upload (wq_weight_data, head_weight_size, vkWq);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          ret = command->upload (wv_weight_data, head_weight_size, vkWv);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          VkTensor Wo (1, attn_output_weight.dim[1], attn_output_weight.dim[0],
                       gpu_, VkTensor::FP16);
          if ((ret = Wo.create ()) != VK_SUCCESS)
            {
              return ret;
            }
          ret = command->upload (
              (__vkllama_fp16_t *)attn_output_weight.weights_data,
              attn_output_weight.num_weights, Wo);

          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          VkTensor vkw1 (1, ffn_gate_weight.dim[1], ffn_gate_weight.dim[0],
                         gpu_, VkTensor::FP16);

          VkTensor vkw2 (1, ffn_down_weight.dim[1], ffn_down_weight.dim[0],
                         gpu_, VkTensor::FP16);

          VkTensor vkw3 (1, ffn_up_weight.dim[1], ffn_up_weight.dim[0], gpu_,
                         VkTensor::FP16);
          if ((ret = vkw1.create ()) != VK_SUCCESS
              || (ret = vkw2.create ()) != VK_SUCCESS
              || (ret = vkw3.create ()) != VK_SUCCESS)
            {
              return ret;
            }

          ret = command->upload (
              (__vkllama_fp16_t *)ffn_gate_weight.weights_data,
              ffn_gate_weight.num_weights, vkw1);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          ret = command->upload (
              (__vkllama_fp16_t *)ffn_down_weight.weights_data,
              ffn_down_weight.num_weights, vkw2);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          ret = command->upload (
              (__vkllama_fp16_t *)ffn_up_weight.weights_data,
              ffn_up_weight.num_weights, vkw3);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          const auto dim = head_dim / head_count;
          Llama2Block::RmsNormParams rmsnorm_params
              = { vk_attn_norm_weight, vk_ffn_norm_weight, norm_eps };
          Llama2Block::TransformerParams transformer_params
              = { vkWk, vkWq, vkWv, Wo, (int)maxlen, (int)dim };
          Llama2Block::FeedForwardParams feedfward_params
              = { vkw1, vkw2, vkw3, norm_eps };

          auto *block = new Llama2Block (gpu_, command, transformer_params,
                                         feedfward_params, rmsnorm_params);

          if ((ret = block->init ()) != VK_SUCCESS)
            {
              return ret;
            }

          blocks_.push_back (block);
          command->end ();
          command->submit ();
        }
    }

    input_command_->wait ();
    output_command_->wait ();
    for (auto c : block_commands_)
      {
        c->wait ();
      }

    return VK_SUCCESS;
  }

  std::vector<uint32_t>
  operator() (std::vector<uint32_t> const &toks, const size_t offset)
  {
    auto t0 = std::chrono::high_resolution_clock::now ();
    VkTensor vktoks (1, 1, toks.size (), gpu_, VkTensor::UINT32, true);
    if (vktoks.create () != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at creating vktoks");
      }

    memcpy (vktoks.host (), toks.data (), sizeof (uint32_t) * toks.size ());
    vktoks.flush ();

    input_command_->begin ();
    VkTensor X = (*input_layer_) (vktoks);
    input_command_->end ();
    input_command_->submit ();

    std::vector<VkTensor> tmps;
    tmps.push_back (X);

    for (int i = 0; i < blocks_.size (); ++i)
      {
        auto *command = block_commands_[i];
        command->begin ();

        auto *block = blocks_[i];
        X = (*block) (X, offset);

        tmps.push_back (X);
        command->end ();
        command->submit ();
      }

    output_command_->begin ();
    VkTensor output = (*output_layer_) (X);
    std::vector<uint32_t> buf (output.size ());
    output_command_->download (output, buf.data (), buf.size ());

    output_command_->end ();
    output_command_->submit ();

    auto t1 = std::chrono::high_resolution_clock::now ();
    input_command_->wait ();
    for (auto *c : block_commands_)
      {
        c->wait ();
      }
    output_command_->wait ();

#ifdef __VKLLAMA_LOG_COST
    auto t2 = std::chrono::high_resolution_clock::now ();
    auto record_cost
        = std::chrono::duration_cast<std::chrono::milliseconds> (t1 - t0)
              .count ();
    auto wait_cost
        = std::chrono::duration_cast<std::chrono::milliseconds> (t2 - t1)
              .count ();

    auto total_cost
        = std::chrono::duration_cast<std::chrono::milliseconds> (t2 - t0)
              .count ();

    fprintf (
        stderr,
        "record cost = %lldms,  wait cost = %lldms, total cost = %lldms\n",
        record_cost, wait_cost, total_cost);
#endif

    for (int i = 0; i < blocks_.size (); ++i)
      {
        blocks_[i]->print_op_cost ();
      }
    return buf;
  }

private:
  GPUDevice *gpu_;
  Command *input_command_;
  std::vector<Command *> block_commands_;
  Command *output_command_;
  InputLayer *input_layer_;
  OutputLayer *output_layer_;
  std::vector<Llama2Block *> blocks_;
};

}

#endif
