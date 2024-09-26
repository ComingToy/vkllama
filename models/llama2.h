#ifndef __VKLLAMA_MODELS_LLAMA2_H__
#define __VKLLAMA_MODELS_LLAMA2_H__
// clang-format off
#include <iterator>
#include <stddef.h>
extern "C"{
#include "gguflib.h"
}
// clang-format on
#include "absl/status/statusor.h"
#include "src/core/command.h"
#include "src/core/common.h"
#include "src/core/tensor.h"
#include "src/ops/argop.h"
#include "src/ops/cast.h"
#include "src/ops/elementwise.h"
#include "src/ops/embedding.h"
#include "src/ops/feed_forward.h"
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
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#define __VKLLAMA_LOG_COST 0
namespace vkllama
{
class InputLayer
{
public:
  InputLayer (GPUDevice *gpu, Command *command, Tensor vocab, uint32_t UNK = 0)
      : gpu_ (gpu), command_ (command), vocab_ (vocab), UNK_ (UNK)
  {
  }

  absl::Status
  init ()
  {
    embedding_op_.reset (new Embedding (gpu_, command_, vocab_, UNK_, FP16));
    auto ret = embedding_op_->init ();
    if (!ret.ok ())
      {
        return ret;
      }

    return absl::OkStatus ();
  }

  absl::StatusOr<Tensor>
  operator() (Tensor toks)
  {
    auto out = embedding_op_->operator() (toks);
    VKLLAMA_STATUS_OK (out);
    return out;
  }

  void
  print_op_cost ()
  {
    fprintf (stderr, "embedding lookup cost -- embedding cost: %llu\n",
             embedding_op_->time ());
  }

private:
  GPUDevice *gpu_;
  Command *command_;
  Tensor vocab_;
  Tensor embs_;

  uint32_t UNK_;
  std::unique_ptr<Embedding> embedding_op_;
};

class Llama2Block
{
public:
  struct TransformerParams
  {
    Tensor Wk;
    Tensor Wq;
    Tensor Wv;
    Tensor Wo;
    int maxlen;
    int dim;
  };

  struct FeedForwardParams
  {
    Tensor w1;
    Tensor w2;
    Tensor w3;
    float eps;
  };

  struct RmsNormParams
  {
    Tensor weight1;
    Tensor weight2;
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

  absl::Status
  init ()
  {
    attn_op_.reset (new MultiHeadAttentionV2 (
        gpu_, command_, transformer_params_.Wk, transformer_params_.Wq,
        transformer_params_.Wv, transformer_params_.Wo,
        transformer_params_.maxlen, transformer_params_.dim, true, FP16,
        true));

    feedforward_op_.reset (new FeedForward (
        gpu_, command_, feedforward_params_.w1, feedforward_params_.w2,
        feedforward_params_.w3, true, FP16));

    norm_op_.reset (new RMSNorm (gpu_, command_, rmsnorm_params_.weight1,
                                 rmsnorm_params_.eps, FP16));
    norm_op2_.reset (new RMSNorm (gpu_, command_, rmsnorm_params_.weight2,
                                  feedforward_params_.eps, FP16));
    add_op_.reset (new ElementWise (gpu_, command_, 0, FP16));
    add_op2_.reset (new ElementWise (gpu_, command_, 0, FP16));

    auto ret = attn_op_->init ();
    if (!ret.ok () || !(ret = feedforward_op_->init ()).ok ()
        || !(ret = norm_op_->init ()).ok ()
        || !(ret = norm_op2_->init ()).ok () || !(ret = add_op_->init ()).ok ()
        || !(ret = add_op2_->init ()).ok ())
      {
        return ret;
      }

    return absl::OkStatus ();
  }

  absl::StatusOr<Tensor>
  operator() (Tensor in, const size_t offset)
  {
    auto ret = norm_op_->operator() (in);

    VKLLAMA_STATUS_OK (ret);
    normed_ = *ret;

    ret = attn_op_->operator() (normed_, offset);
    VKLLAMA_STATUS_OK (ret);
    transformed_ = *ret;

    ret = add_op_->operator() (transformed_, in);
    VKLLAMA_STATUS_OK (ret);
    added_ = *ret;

    ret = norm_op2_->operator() (added_);
    VKLLAMA_STATUS_OK (ret);
    normed2_ = *ret;

    ret = feedforward_op_->operator() (normed2_);
    VKLLAMA_STATUS_OK (ret);
    feed_ = *ret;

    auto out = add_op2_->operator() (feed_, added_);
    VKLLAMA_STATUS_OK (out);

    return out;
  }

  void
  print_op_cost ()
  {
    fprintf (stderr,
             "block cost -- attn norm cost: %llu, attn cost: %lld, attn add "
             "cost: %lld, "
             "fffn norm cost: %lld, ffn cost: %lld, ffn add cost: %lld\n",
             norm_op_->time (), attn_op_->time (), 0llu,
             feedforward_op_->time (), 0llu, 0llu);
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

  Tensor normed_;
  Tensor normed2_;
  Tensor transformed_;
  Tensor added_;
  Tensor feed_;
};

class OutputLayer
{
public:
  OutputLayer (GPUDevice *gpu, Command *command, Tensor wo, Tensor norm_weight)
      : gpu_ (gpu), command_ (command), wo_ (wo), norm_weight_ (norm_weight)
  {
  }

  absl::Status
  init ()
  {
    matmul_op_.reset (
        new MatMul (gpu_, command_, wo_, 1.0, .0, 0, 0, true, FP16));
    auto ret = matmul_op_->init ();
    if (!ret.ok ())
      {
        return ret;
      }

    norm_op_.reset (new RMSNorm (gpu_, command_, norm_weight_, 1e-6, FP16));
    if (!(ret = norm_op_->init ()).ok ())
      {
        return ret;
      }

    cast_op_.reset (new Cast (gpu_, command_, FP16, FP32));
    if (!(ret = cast_op_->init ()).ok ())
      {
        return ret;
      }
    argmax_op_.reset (new ArgMax (gpu_, command_, FP16));
    return argmax_op_->init ();
  }

  absl::StatusOr<Tensor>
  operator() (Tensor in)
  {
    auto ret = norm_op_->operator() (in);
    VKLLAMA_STATUS_OK (ret);
    norm_output_ = *ret;

    ret = matmul_op_->operator() (norm_output_);
    VKLLAMA_STATUS_OK (ret);
    matmul_output_ = *ret;

    auto out = (*cast_op_) (matmul_output_);
    VKLLAMA_STATUS_OK (out);

    return out;
  }

  void
  print_op_cost ()
  {
    fprintf (stderr,
             "output cost -- matmul out cost: %llu, norm cost: %llu, argmax "
             "cost: %llu\n",
             matmul_op_->time (), norm_op_->time (), argmax_op_->time ());
  }

private:
  GPUDevice *gpu_;
  Command *command_;
  Tensor wo_;
  Tensor norm_weight_;
  Tensor norm_output_;
  Tensor matmul_output_;
  std::unique_ptr<MatMul> matmul_op_;
  std::unique_ptr<RMSNorm> norm_op_;
  std::unique_ptr<ArgMax> argmax_op_;
  std::unique_ptr<Cast> cast_op_;
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

  absl::Status
  init (std::map<std::string, gguf_key> &kv,
        std::map<std::string, gguf_tensor> &tensors)
  {
    gpu_ = new GPUDevice ();
    auto ret = gpu_->init ();
    if (!ret.ok ())
      {
        return ret;
      }

    input_command_ = new Command (gpu_);
    output_command_ = new Command (gpu_);
    if (!(ret = input_command_->init ()).ok ()
        || !(ret = output_command_->init ()).ok ())
      {
        return ret;
      }

    auto head_count = kv["llama.attention.head_count"].val->uint32;
    auto block_count = kv["llama.block_count"].val->uint32;
    auto norm_eps = kv["llama.attention.layer_norm_rms_epsilon"].val->float32;
    auto maxlen = kv["llama.context_length"].val->uint32;

    if (ret = input_command_->begin (); !ret.ok ())
      {
        return ret;
      }

    if (ret = output_command_->begin (); !ret.ok ())
      {
        return ret;
      }

    auto to_dtype = [] (const uint32_t type) {
      if (type == 0)
        {
          return FP32;
        }
      if (type == 1)
        {
          return FP16;
        }
      if (type == 8)
        {
          return Q8_0;
        }

      return INT8;
    };

    // input layer
    {
      auto embeddings = tensors["token_embd.weight"];
      auto output_weight = tensors["output.weight"];
      auto norm_weight = tensors["output_norm.weight"];

      Tensor vkembeddings (1, embeddings.dim[1], embeddings.dim[0], gpu_,
                           to_dtype (embeddings.type));
      Tensor vkoutput_weight (1, output_weight.dim[1], output_weight.dim[0],
                              gpu_, to_dtype (output_weight.type));
      Tensor vknorm_weight (1, 1, norm_weight.dim[0], gpu_,
                            to_dtype (norm_weight.type));

      absl::Status ret;

      if (!(ret = vkembeddings.create ()).ok ()
          || !(ret = vkoutput_weight.create ()).ok ()
          || !(ret = vknorm_weight.create ()).ok ())
        {
          return ret;
        }

      ret = input_command_->upload ((const uint8_t *)embeddings.weights_data,
                                    embeddings.bsize, vkembeddings);
      if (!ret.ok ())
        {
          return ret;
        }

      ret = output_command_->upload (
          (const uint8_t *)output_weight.weights_data, output_weight.bsize,
          vkoutput_weight);
      if (!ret.ok ())
        {
          return ret;
        }

      ret = input_command_->upload ((const uint8_t *)norm_weight.weights_data,
                                    norm_weight.bsize, vknorm_weight);
      if (!ret.ok ())
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

      if (!(ret = input_layer_->init ()).ok ()
          || !(ret = output_layer_->init ()).ok ())
        {
          return ret;
        }

      if (ret = input_command_->end (); !ret.ok ())
        {
          return ret;
        }

      if (ret = output_command_->end (); !ret.ok ())
        {
          return ret;
        }

      if (ret = input_command_->submit_and_wait (); !ret.ok ())
        {
          return ret;
        }

      if (ret = output_command_->submit_and_wait (); !ret.ok ())
        {
          return ret;
        }
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
          if (!(ret = command->init ()).ok ()
              || !(ret = command->begin ()).ok ())
            {
              return ret;
            }

          block_commands_.push_back (command);

          Tensor vk_attn_norm_weight (1, 1, attn_norm_weight.dim[0], gpu_,
                                      to_dtype (attn_norm_weight.type));

          Tensor vk_ffn_norm_weight (1, 1, ffn_norm_weight.dim[0], gpu_,
                                     to_dtype (ffn_norm_weight.type));

          absl::Status ret;

          if (!(ret = vk_attn_norm_weight.create ()).ok ()
              || !(ret = vk_ffn_norm_weight.create ()).ok ())
            {
              return ret;
            }

          ret = command->upload (
              (const uint8_t *)attn_norm_weight.weights_data,
              attn_norm_weight.bsize, vk_attn_norm_weight);
          if (!ret.ok ())
            {
              return ret;
            }

          ret = command->upload ((const uint8_t *)ffn_norm_weight.weights_data,
                                 ffn_norm_weight.bsize, vk_ffn_norm_weight);
          if (!ret.ok ())
            {
              return ret;
            }

          size_t head_dim = attn_k_weight.dim[1];
          size_t input_dim = attn_k_weight.dim[0];

          Tensor vkWk (1, head_dim, input_dim, gpu_,
                       to_dtype (attn_k_weight.type));
          Tensor vkWq (1, head_dim, input_dim, gpu_,
                       to_dtype (attn_q_weight.type));
          Tensor vkWv (1, head_dim, input_dim, gpu_,
                       to_dtype (attn_v_weight.type));

          if (!(ret = vkWk.create ()).ok () || !(ret = vkWq.create ()).ok ()
              || !(ret = vkWv.create ()).ok ())
            {
              return ret;
            }

          const auto *wk_weight_data
              = (const uint8_t *)attn_k_weight.weights_data;

          const auto *wq_weight_data
              = (const uint8_t *)attn_q_weight.weights_data;

          const auto *wv_weight_data
              = (const uint8_t *)attn_v_weight.weights_data;

          ret = command->upload (wk_weight_data, attn_k_weight.bsize, vkWk);
          if (!ret.ok ())
            {
              return ret;
            }

          ret = command->upload (wq_weight_data, attn_q_weight.bsize, vkWq);
          if (!ret.ok ())
            {
              return ret;
            }

          ret = command->upload (wv_weight_data, attn_v_weight.bsize, vkWv);
          if (!ret.ok ())
            {
              return ret;
            }

          Tensor Wo (1, attn_output_weight.dim[1], attn_output_weight.dim[0],
                     gpu_, to_dtype (attn_output_weight.type));
          if (!(ret = Wo.create ()).ok ())
            {
              return ret;
            }
          ret = command->upload (
              (const uint8_t *)attn_output_weight.weights_data,
              attn_output_weight.bsize, Wo);

          if (!ret.ok ())
            {
              return ret;
            }

          Tensor vkw1 (1, ffn_gate_weight.dim[1], ffn_gate_weight.dim[0], gpu_,
                       to_dtype (ffn_gate_weight.type));

          Tensor vkw2 (1, ffn_down_weight.dim[1], ffn_down_weight.dim[0], gpu_,
                       to_dtype (ffn_down_weight.type));

          Tensor vkw3 (1, ffn_up_weight.dim[1], ffn_up_weight.dim[0], gpu_,
                       to_dtype (ffn_up_weight.type));

          if (!(ret = vkw1.create ()).ok () || !(ret = vkw2.create ()).ok ()
              || !(ret = vkw3.create ()).ok ())
            {
              return ret;
            }

          ret = command->upload ((const uint8_t *)ffn_gate_weight.weights_data,
                                 ffn_gate_weight.bsize, vkw1);

          if (!ret.ok ())
            {
              return ret;
            }

          ret = command->upload ((const uint8_t *)ffn_down_weight.weights_data,
                                 ffn_down_weight.bsize, vkw2);

          if (!ret.ok ())
            {
              return ret;
            }

          ret = command->upload ((const uint8_t *)ffn_up_weight.weights_data,
                                 ffn_up_weight.bsize, vkw3);

          if (!ret.ok ())
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

          if (!(ret = block->init ()).ok ())
            {
              return ret;
            }

          blocks_.push_back (block);

          if (ret = command->end (); !ret.ok ())
            {
              return ret;
            }

          if (ret = command->submit_and_wait (); !ret.ok ())
            {
              return ret;
            }
        }
    }

    return absl::OkStatus ();
  }

  absl::StatusOr<std::vector<float> >
  operator() (std::vector<uint32_t> const &toks, const size_t offset)
  {
    auto t0 = std::chrono::high_resolution_clock::now ();
    Tensor vktoks (1, 1, toks.size (), gpu_, UINT32, true);
    if (!vktoks.create ().ok ())
      {
        throw std::runtime_error ("failed at creating vktoks");
      }

    memcpy (vktoks.host (), toks.data (), sizeof (uint32_t) * toks.size ());

    if (auto ret = vktoks.flush (); !ret.ok ())
      {
        throw std::runtime_error (ret.ToString ());
      }

    if (auto ret = input_command_->begin (); !ret.ok ())
      {
        throw std::runtime_error (ret.ToString ());
      }

    auto X = (*input_layer_) (vktoks);
    if (!X.ok ())
      {
        return X.status ();
      }

    if (auto ret = input_command_->end (); !ret.ok ())
      {
        return ret;
      }

    if (auto ret = input_command_->submit (); !ret.ok ())
      {
        return ret;
      }

    std::vector<Tensor> tmps;
    tmps.push_back (*X);

    for (int i = 0; i < blocks_.size (); ++i)
      {
        auto *command = block_commands_[i];

        if (auto ret = command->begin (); !ret.ok ())
          {
            throw std::runtime_error (ret.ToString ());
          }

        auto *block = blocks_[i];
        X = (*block) (*X, offset);
        if (!X.ok ())
          {
            return X.status ();
          }

        tmps.push_back (*X);

        if (auto ret = command->end (); !ret.ok ())
          {
            throw std::runtime_error (ret.ToString ());
          }

        if (auto ret = command->submit (); !ret.ok ())
          {
            throw std::runtime_error (ret.ToString ());
          }
      }

    if (auto ret = output_command_->begin (); !ret.ok ())
      {
        throw std::runtime_error (ret.ToString ());
      }

    auto output = (*output_layer_) (*X);

    std::vector<float> buf_logits;
    buf_logits.resize (output->size ());

    auto ret = output_command_->download (*output, buf_logits.data (),
                                          buf_logits.size ());
    if (!ret.ok ())
      {
        return ret;
      }

    ret = output_command_->end ();
    if (!ret.ok ())
      {
        return ret;
      }

    ret = output_command_->submit ();
    if (!ret.ok ())
      {
        return ret;
      }

    auto t1 = std::chrono::high_resolution_clock::now ();
    ret = input_command_->wait ();
    if (!ret.ok ())
      {
        return ret;
      }

    for (auto *c : block_commands_)
      {
        ret = c->wait ();
        if (!ret.ok ())
          {
            return ret;
          }
      }

    ret = output_command_->wait ();
    if (!ret.ok ())
      {
        return ret;
      }

#if __VKLLAMA_LOG_COST
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

    input_layer_->print_op_cost ();
    for (int i = 0; i < blocks_.size (); ++i)
      {
        blocks_[i]->print_op_cost ();
      }
    output_layer_->print_op_cost ();

#endif
    return buf_logits;
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
