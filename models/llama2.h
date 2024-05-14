#ifndef __VKLLAMA_MODELS_LLAMA2_H__
#define __VKLLAMA_MODELS_LLAMA2_H__
#include "models/gguf/gguf.h"
#include "models/proto/llama2_model.pb.h"
#include "src/core/command.h"
#include "src/core/tensor.h"
#include "src/ops/argop.h"
#include "src/ops/elementwise.h"
#include "src/ops/embedding.h"
#include "src/ops/feed_forward.h"
#include "src/ops/multiheadattention.h"
#include "src/ops/rms_norm.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

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
    embedding_op_.reset (new Embedding (gpu_, command_, vocab_, UNK_));
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
    VkTensor weight1;
    VkTensor weight2;
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
    norm_op_.reset (new RMSNorm (gpu_, command_, rmsnorm_params_.weight1));
    norm_op2_.reset (new RMSNorm (gpu_, command_, rmsnorm_params_.weight2));
    add_op_.reset (new ElementWise (gpu_, command_, 0));

    auto ret = attn_op_->init ();
    if (ret != VK_SUCCESS || (ret = feedforward_op_->init ()) != VK_SUCCESS
        || (ret = norm_op_->init ()) != VK_SUCCESS
        || (ret = norm_op2_->init ()) != VK_SUCCESS
        || (ret = add_op_->init ()) != VK_SUCCESS)
      {
        return ret;
      }

    return VK_SUCCESS;
  }

  VkTensor
  operator() (VkTensor in)
  {
    auto ret = norm_op_->operator() (in, normed_);
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
    ret = feedforward_op_->operator() (normed2_, out);
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
  std::unique_ptr<RMSNorm> norm_op2_;
  std::unique_ptr<ElementWise> add_op_;

  TransformerParams transformer_params_;
  FeedForwardParams feedforward_params_;
  RmsNormParams rmsnorm_params_;

  VkTensor normed_;
  VkTensor normed2_;
  VkTensor transformed_;
  VkTensor added_;
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
    auto ret = matmul_op_->init ();
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    argmax_op_.reset (new ArgMax (gpu_, command_));
    return argmax_op_->init ();
  }

  VkTensor
  operator() (VkTensor in)
  {
    auto ret = matmul_op_->operator() (in, wo_, matmul_output_);
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
  VkTensor matmul_output_;
  std::unique_ptr<MatMul> matmul_op_;
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

    for (auto *command : block_commands_)
      {
        delete command;
      }
    delete input_command_;
    delete output_command_;
    delete gpu_;
  }

  VkResult
  init (std::string const &path)
  {
    GGUF gguf (path);
    if (gguf.init () != 0)
      {
        return VK_ERROR_UNKNOWN;
      }

    gpu_ = new GPUDevice ();
    gpu_->init ();
    input_command_ = new Command (gpu_);
    output_command_ = new Command (gpu_);
    input_command_->init ();
    output_command_->init ();

    uint32_t head_count;
    uint32_t head_count_kv;
    uint32_t block_count;
    if (gguf.get ("llama.attention.head_count", head_count) != 0
        || gguf.get ("llama.attention.head_count_kv", head_count_kv) != 0
        || gguf.get ("llama.block_count", block_count) != 0)
      {
        return VK_ERROR_UNKNOWN;
      }

    for (uint32_t i = 0; i < block_count; ++i)
      {
        block_commands_.push_back (new Command (gpu_));
        block_commands_.back ()->init ();
      }

    input_command_->begin ();
    // input layer
    {
      const auto *embeddings = gguf.get_tensor ("token_embd.weight");
      VkTensor vkembeddings (1, embeddings->info->dims[1],
                             embeddings->info->dims[0], gpu_);

      VkResult ret = VK_SUCCESS;
      if ((ret = vkembeddings.create ()) != VK_SUCCESS)
        {
          return ret;
        }

      ret = input_command_->upload (
          (const float *)embeddings->data,
          embeddings->info->dims[0] * embeddings->info->dims[1], vkembeddings);
      if (ret != VK_SUCCESS)
        {
          return ret;
        }

      if (ret != VK_SUCCESS)
        {
          return ret;
        }

      {
        uint32_t UNK = 0;
        if (gguf.get ("tokenizer.ggml.padding_token_id", UNK) != 0)
          {
            return VK_ERROR_UNKNOWN;
          }

        input_layer_
            = new InputLayer (gpu_, input_command_, vkembeddings, UNK);
      }

      output_layer_ = new OutputLayer (gpu_, output_command_, vkembeddings);

      if ((ret = input_layer_->init ()) != VK_SUCCESS
          || (ret = output_layer_->init ()) != VK_SUCCESS)
        {
          return ret;
        }
    }

    input_command_->end ();
    input_command_->submit ();

    // blocks
    {

      char vname[512];
      for (uint32_t b = 0; b < block_count; ++b)
        {
          auto command_ = block_commands_[b];
          command_->begin ();
          ::snprintf (vname, sizeof (vname), "blk.%u.attn_norm.weight", b);
          auto const *attn_norm_weight = gguf.get_tensor (vname);

          ::snprintf (vname, sizeof (vname), "blk.%u.attn_k.weight", b);
          auto attn_k_weight = gguf.get_tensor (vname);

          ::snprintf (vname, sizeof (vname), "blk.%u.attn_k.weight", b);
          auto attn_q_weight = gguf.get_tensor (vname);

          ::snprintf (vname, sizeof (vname), "blk.%u.attn_k.weight", b);
          auto attn_v_weight = gguf.get_tensor (vname);

          ::snprintf (vname, sizeof (vname), "blk.%u.attn_output.weight", b);
          const auto *attn_output_weight = gguf.get_tensor (vname);

          ::snprintf (vname, sizeof (vname), "blk.%u.ffn_norm.weight", b);
          const auto *ffn_norm_weight = gguf.get_tensor (vname);

          ::snprintf (vname, sizeof (vname), "blk.%u.ffn_up.weight", b);
          const auto *ffn_up_weight = gguf.get_tensor (vname);

          ::snprintf (vname, sizeof (vname), "blk.%u.ffn_down.weight", b);
          const auto *ffn_down_weight = gguf.get_tensor (vname);

          ::snprintf (vname, sizeof (vname), "blk.%u.ffn_gate.weight", b);
          const auto *ffn_gate_weight = gguf.get_tensor (vname);

          if (!attn_norm_weight || !attn_k_weight || !attn_q_weight
              || !attn_v_weight || !attn_output_weight || !ffn_norm_weight
              || !ffn_up_weight || !ffn_down_weight || !ffn_gate_weight)
            {
              return VK_ERROR_UNKNOWN;
            }

          VkTensor vk_attn_norm_weight (1, 1, attn_norm_weight->info->dims[0],
                                        gpu_);

          VkTensor vk_ffn_norm_weight (1, 1, ffn_norm_weight->info->dims[0],
                                       gpu_);

          VkResult ret = VK_SUCCESS;
          if ((ret = vk_attn_norm_weight.create ()) != VK_SUCCESS
              || (ret = vk_ffn_norm_weight.create ()) != VK_SUCCESS)
            {
              return ret;
            }

          ret = command_->upload ((const float *)attn_norm_weight->data,
                                  attn_norm_weight->info->dims[0],
                                  vk_attn_norm_weight);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          ret = command_->upload ((const float *)ffn_norm_weight->data,
                                  ffn_norm_weight->info->dims[0],
                                  vk_ffn_norm_weight);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          std::vector<VkTensor> Wk, Wq, Wv;
          size_t head_dim = attn_k_weight->info->dims[0] / head_count;
          size_t input_dim = attn_k_weight->info->dims[1];
          size_t head_weight_size = head_dim * input_dim;

          for (int h = 0; h < head_count; ++h)
            {
              VkTensor vkWk (1, head_dim, input_dim, gpu_);
              VkTensor vkWq (1, head_dim, input_dim, gpu_);
              VkTensor vkWv (1, head_dim, input_dim, gpu_);

              if ((ret = vkWk.create ()) != VK_SUCCESS
                  || (ret = vkWq.create ()) != VK_SUCCESS
                  || (ret = vkWv.create ()) != VK_SUCCESS)
                {
                  return ret;
                }

              const float *wk_weight_data
                  = (const float *)attn_k_weight->data + head_weight_size * h;
              const float *wq_weight_data
                  = (const float *)attn_q_weight->data + head_weight_size * h;
              const float *wv_weight_data
                  = (const float *)attn_v_weight->data + head_weight_size * h;

              ret = command_->upload (wk_weight_data, head_weight_size, vkWk);
              if (ret != VK_SUCCESS)
                {
                  return ret;
                }

              ret = command_->upload (wq_weight_data, head_weight_size, vkWq);
              if (ret != VK_SUCCESS)
                {
                  return ret;
                }

              ret = command_->upload (wv_weight_data, head_weight_size, vkWv);
              if (ret != VK_SUCCESS)
                {
                  return ret;
                }

              Wk.push_back (vkWk);
              Wq.push_back (vkWq);
              Wv.push_back (vkWv);
            }

          VkTensor Wo (1, attn_output_weight->info->dims[0],
                       attn_output_weight->info->dims[1], gpu_);
          if ((ret = Wo.create ()) != VK_SUCCESS)
            {
              return ret;
            }
          ret = command_->upload ((const float *)attn_output_weight->data,
                                  attn_output_weight->info->dims[0]
                                      * attn_output_weight->info->dims[1],
                                  Wo);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          VkTensor vkw1 (1, ffn_up_weight->info->dims[0],
                         ffn_up_weight->info->dims[1], gpu_);

          VkTensor vkw2 (1, ffn_down_weight->info->dims[0],
                         ffn_down_weight->info->dims[1], gpu_);
          VkTensor vkw3 (1, ffn_gate_weight->info->dims[0],
                         ffn_gate_weight->info->dims[1], gpu_);
          if ((ret = vkw1.create ()) != VK_SUCCESS
              || (ret = vkw2.create ()) != VK_SUCCESS
              || (ret = vkw3.create ()) != VK_SUCCESS)
            {
              return ret;
            }

          ret = command_->upload ((const float *)ffn_up_weight->data,
                                  ffn_up_weight->info->dims[0]
                                      * ffn_up_weight->info->dims[1],
                                  vkw1);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          ret = command_->upload ((const float *)ffn_down_weight->data,
                                  ffn_down_weight->info->dims[0]
                                      * ffn_down_weight->info->dims[1],
                                  vkw2);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          ret = command_->upload ((const float *)ffn_gate_weight->data,
                                  ffn_gate_weight->info->dims[0]
                                      * ffn_gate_weight->info->dims[1],
                                  vkw3);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          Llama2Block::RmsNormParams rmsnorm_params
              = { vk_attn_norm_weight, vk_ffn_norm_weight };
          Llama2Block::TransformerParams transformer_params
              = { Wk, Wq, Wv, Wo, 1024, 512 };
          Llama2Block::FeedForwardParams feedfward_params
              = { vkw1, vkw2, vkw3 };

          auto *block = new Llama2Block (gpu_, command_, transformer_params,
                                         feedfward_params, rmsnorm_params);

          if ((ret = block->init ()) != VK_SUCCESS)
            {
              return ret;
            }

          blocks_.push_back (block);
          command_->end ();
          command_->submit ();
        }
    }

    input_command_->wait ();
    for (auto *c : block_commands_)
      {
        c->wait ();
      }
    return VK_SUCCESS;
  }

  std::vector<uint32_t>
  operator() (std::vector<uint32_t> const &toks)
  {
    auto t0 = std::chrono::high_resolution_clock::now ();
    input_command_->begin ();
    VkTensor vktoks (1, 1, toks.size (), gpu_, VkTensor::UINT32);
    if (vktoks.create () != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at creating vktoks");
      }
    auto ret = input_command_->upload (toks.data (), toks.size (), vktoks);
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at uploading toks");
      }

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
        X = (*block) (X);
        tmps.push_back (X);
        command->end ();
        command->submit ();
      }

    output_command_->begin ();
    VkTensor output = (*output_layer_) (X);
    std::vector<uint32_t> buf (output.size ());

    ret = output_command_->download (output, buf.data (), buf.size ());
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at downloadding outputs");
      }

    ret = output_command_->end ();
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at end commands");
      }
    output_command_->submit ();

    auto t1 = std::chrono::high_resolution_clock::now ();
    input_command_->wait ();
    for (auto *c : block_commands_)
      {
        c->wait ();
      }
    ret = output_command_->wait ();
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at submit");
      }
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

#endif
