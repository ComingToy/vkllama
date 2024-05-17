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
#include <unordered_map>
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
        transformer_params_.maxlen, transformer_params_.dim, true));
    feedforward_op_.reset (new FeedForward (
        gpu_, command_, feedforward_params_.w1, feedforward_params_.w2,
        feedforward_params_.w3, true));
    norm_op_.reset (new RMSNorm (gpu_, command_, rmsnorm_params_.weight1));
    norm_op2_.reset (new RMSNorm (gpu_, command_, rmsnorm_params_.weight2));
    add_op_.reset (new ElementWise (gpu_, command_, 0));
    add_op2_.reset (new ElementWise (gpu_, command_, 0));

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

private:
  GPUDevice *gpu_;
  Command *command_;
  std::unique_ptr<MultiHeadAttention> attn_op_;
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
    matmul_op_.reset (new MatMul (gpu_, command_, wo_, 0, 0, true));
    auto ret = matmul_op_->init ();
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    norm_op_.reset (new RMSNorm (gpu_, command_, norm_weight_));
    if ((ret = norm_op_->init ()) != VK_SUCCESS)
      {
        return ret;
      }

    argmax_op_.reset (new ArgMax (gpu_, command_));
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

    return matmul_output_;

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
  Model () : gpu_ (nullptr), command_ (nullptr) {}

  ~Model ()
  {
    delete input_layer_;
    delete output_layer_;
    for (auto *block : blocks_)
      {
        delete block;
      }

    delete command_;
    delete gpu_;
  }

  VkResult
  init (std::unordered_map<std::string, const llama2::Variable *> &variables)
  {
    gpu_ = new GPUDevice ();
    gpu_->init ();
    command_ = new Command (gpu_);
    command_->init ();

    uint32_t head_count = 32;
    uint32_t block_count = 26;

    command_->begin ();
    // input layer
    {
      const auto *embeddings = variables["model.embed_tokens.weight"];
      const auto *output_weight = variables["lm_head.weight"];
      const auto *norm_weight = variables["model.norm.weight"];

      VkTensor vkembeddings (1, embeddings->shape (0), embeddings->shape (1),
                             gpu_);
      VkTensor vkoutput_weight (1, output_weight->shape (0),
                                output_weight->shape (1), gpu_);
      VkTensor vknorm_weight (1, 1, norm_weight->shape (0), gpu_);

      VkResult ret = VK_SUCCESS;
      if ((ret = vkembeddings.create ()) != VK_SUCCESS
          || (ret = vkoutput_weight.create ()) != VK_SUCCESS
          || (ret = vknorm_weight.create ()) != VK_SUCCESS)
        {
          return ret;
        }

      ret = command_->upload (embeddings->f32_values ().data (),
                              embeddings->f32_values_size (), vkembeddings);
      if (ret != VK_SUCCESS)
        {
          return ret;
        }

      ret = command_->upload (output_weight->f32_values ().data (),
                              output_weight->f32_values_size (),
                              vkoutput_weight);
      if (ret != VK_SUCCESS)
        {
          return ret;
        }

      ret = command_->upload (norm_weight->f32_values ().data (),
                              norm_weight->f32_values_size (), vknorm_weight);
      if (ret != VK_SUCCESS)
        {
          return ret;
        }

      {
        uint32_t UNK = 0;
        input_layer_ = new InputLayer (gpu_, command_, vkembeddings, UNK);
      }

      output_layer_
          = new OutputLayer (gpu_, command_, vkoutput_weight, vknorm_weight);

      if ((ret = input_layer_->init ()) != VK_SUCCESS
          || (ret = output_layer_->init ()) != VK_SUCCESS)
        {
          return ret;
        }
    }

    // blocks
    {

      char vname[512];
      for (uint32_t b = 0; b < block_count; ++b)
        {
          ::snprintf (vname, sizeof (vname),
                      "model.layers.%u.input_layernorm.weight", b);
          auto const *attn_norm_weight = variables[vname];

          ::snprintf (vname, sizeof (vname),
                      "model.layers.%u.self_attn.k_proj.weight", b);
          auto attn_k_weight = variables[vname];

          ::snprintf (vname, sizeof (vname),
                      "model.layers.%u.self_attn.q_proj.weight", b);
          auto attn_q_weight = variables[vname];

          ::snprintf (vname, sizeof (vname),
                      "model.layers.%u.self_attn.v_proj.weight", b);
          auto attn_v_weight = variables[vname];

          ::snprintf (vname, sizeof (vname),
                      "model.layers.%u.self_attn.o_proj.weight", b);
          const auto *attn_output_weight = variables[vname];

          ::snprintf (vname, sizeof (vname),
                      "model.layers.%u.post_attention_layernorm.weight", b);
          const auto *ffn_norm_weight = variables[vname];

          ::snprintf (vname, sizeof (vname),
                      "model.layers.%u.mlp.up_proj.weight", b);
          const auto *ffn_up_weight = variables[vname];

          ::snprintf (vname, sizeof (vname),
                      "model.layers.%u.mlp.down_proj.weight", b);
          const auto *ffn_down_weight = variables[vname];

          ::snprintf (vname, sizeof (vname),
                      "model.layers.%u.mlp.gate_proj.weight", b);

          const auto *ffn_gate_weight = variables[vname];

          if (!attn_norm_weight || !attn_k_weight || !attn_q_weight
              || !attn_v_weight || !attn_output_weight || !ffn_norm_weight
              || !ffn_up_weight || !ffn_down_weight || !ffn_gate_weight)
            {
              return VK_ERROR_UNKNOWN;
            }

          VkTensor vk_attn_norm_weight (1, 1, attn_norm_weight->shape (0),
                                        gpu_);

          VkTensor vk_ffn_norm_weight (1, 1, ffn_norm_weight->shape (0), gpu_);

          VkResult ret = VK_SUCCESS;
          if ((ret = vk_attn_norm_weight.create ()) != VK_SUCCESS
              || (ret = vk_ffn_norm_weight.create ()) != VK_SUCCESS)
            {
              return ret;
            }

          ret = command_->upload (attn_norm_weight->f32_values ().data (),
                                  attn_norm_weight->f32_values_size (),
                                  vk_attn_norm_weight);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          ret = command_->upload (ffn_norm_weight->f32_values ().data (),
                                  ffn_norm_weight->f32_values_size (),
                                  vk_ffn_norm_weight);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          std::vector<VkTensor> Wk, Wq, Wv;
          size_t head_dim = attn_k_weight->shape (0) / head_count;
          size_t input_dim = attn_k_weight->shape (1);
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
                  = attn_k_weight->f32_values ().data ()
                    + head_weight_size * h;
              const float *wq_weight_data
                  = attn_q_weight->f32_values ().data ()
                    + head_weight_size * h;
              const float *wv_weight_data
                  = attn_v_weight->f32_values ().data ()
                    + head_weight_size * h;

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

          VkTensor Wo (1, attn_output_weight->shape (0),
                       attn_output_weight->shape (1), gpu_);
          if ((ret = Wo.create ()) != VK_SUCCESS)
            {
              return ret;
            }
          ret = command_->upload (attn_output_weight->f32_values ().data (),
                                  attn_output_weight->f32_values_size (), Wo);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          VkTensor vkw1 (1, ffn_gate_weight->shape (0),
                         ffn_gate_weight->shape (1), gpu_);

          VkTensor vkw2 (1, ffn_down_weight->shape (0),
                         ffn_down_weight->shape (1), gpu_);

          VkTensor vkw3 (1, ffn_up_weight->shape (0), ffn_up_weight->shape (1),
                         gpu_);
          if ((ret = vkw1.create ()) != VK_SUCCESS
              || (ret = vkw2.create ()) != VK_SUCCESS
              || (ret = vkw3.create ()) != VK_SUCCESS)
            {
              return ret;
            }

          ret = command_->upload (ffn_gate_weight->f32_values ().data (),
                                  ffn_gate_weight->f32_values_size (), vkw1);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          ret = command_->upload (ffn_down_weight->f32_values ().data (),
                                  ffn_down_weight->f32_values_size (), vkw2);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          ret = command_->upload (ffn_up_weight->f32_values ().data (),
                                  ffn_up_weight->f32_values_size (), vkw3);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          Llama2Block::RmsNormParams rmsnorm_params
              = { vk_attn_norm_weight, vk_ffn_norm_weight };
          Llama2Block::TransformerParams transformer_params
              = { Wk, Wq, Wv, Wo, 1024, (int)head_dim };
          Llama2Block::FeedForwardParams feedfward_params
              = { vkw1, vkw2, vkw3 };

          auto *block = new Llama2Block (gpu_, command_, transformer_params,
                                         feedfward_params, rmsnorm_params);

          if ((ret = block->init ()) != VK_SUCCESS)
            {
              return ret;
            }

          blocks_.push_back (block);
        }
    }

    command_->end ();
    command_->submit ();
    command_->wait ();
    return VK_SUCCESS;
  }

  std::vector<uint32_t>
  operator() (std::vector<uint32_t> const &toks)
  {
    auto t0 = std::chrono::high_resolution_clock::now ();
    command_->begin ();
    VkTensor vktoks (1, 1, toks.size (), gpu_, VkTensor::UINT32);
    if (vktoks.create () != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at creating vktoks");
      }
    auto ret = command_->upload (toks.data (), toks.size (), vktoks);
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at uploading toks");
      }

    VkTensor X = (*input_layer_) (vktoks);

    std::vector<float> emb_buf (X.size ());
    command_->download (X, emb_buf.data (), emb_buf.size ());

    std::vector<VkTensor> tmps;
    tmps.push_back (X);

    std::vector<std::vector<float> > block_bufs (blocks_.size ());
    for (int i = 0; i < blocks_.size (); ++i)
      {
        auto *block = blocks_[i];
        X = (*block) (X);
        tmps.push_back (X);
        block_bufs[i].resize (X.size ());
        command_->download (X, block_bufs[i].data (), block_bufs[i].size ());
      }

    VkTensor output = (*output_layer_) (X);
    std::vector<float> logits (output.size ());
    command_->download (output, logits.data (), logits.size ());
#if 0
    ret = output_command_->download (output, buf.data (), buf.size ());
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at downloadding outputs");
      }
#endif

    command_->end ();
    command_->submit ();

    auto t1 = std::chrono::high_resolution_clock::now ();
    command_->wait ();

    size_t dim = logits.size () / toks.size ();
    std::vector<uint32_t> buf;
    for (int h = 0; h < toks.size (); ++h)
      {
        int i = 0;
        float v = logits[h * dim];
        float sum = .0;
        for (int w = 0; w < dim; ++w)
          {
            sum += logits[h * dim + w];
            if (logits[h * dim + w] > v)
              {
                i = w;
                v = logits[h * dim + w];
              }
          }
        fprintf (stderr, "mean of %d logits: %f, argmax = %d\n", h,
                 sum / (float)dim, i);
        buf.push_back (i);
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
  // Command *input_command_;
  // std::vector<Command *> block_commands_;
  // Command *output_command_;
  Command *command_;
  InputLayer *input_layer_;
  OutputLayer *output_layer_;
  std::vector<Llama2Block *> blocks_;
};

#endif
