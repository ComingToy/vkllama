#ifndef __VKLLAMA_MODELS_LLAMA2_H__
#define __VKLLAMA_MODELS_LLAMA2_H__
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
              VkTensor weight, uint32_t UNK = 0)
      : gpu_ (gpu), command_ (command), vocab_ (vocab), norm_weights_ (weight),
        UNK_ (UNK)
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

    norm_op_.reset (new RMSNorm (gpu_, command_, norm_weights_));
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
    auto ret = embedding_op_->operator() (toks, embs_);
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at forwarding embedding op");
      }

    out = embs_;
    if ((ret = norm_op_->operator() (embs_, out)) != VK_SUCCESS)
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
  {
    gpu_ = new GPUDevice ();
    command_ = new Command (gpu_);
    gpu_->init ();
    command_->init ();
  }

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
  init (std::string const &path)
  {
    // auto ret = gpu_->init ();
    // if (ret != VK_SUCCESS)
    //   {
    //     return ret;
    //   }

    // if ((ret = command_->init ()) != VK_SUCCESS)
    //   {
    //     return ret;
    //   }

    VkResult ret = VK_SUCCESS;
    llama2::Variables chkpts;
    int fd = -1;
    if ((fd = ::open (path.c_str (), O_RDONLY)) < 0)
      {
        return VK_ERROR_UNKNOWN;
      }

    if (!chkpts.ParseFromFileDescriptor (fd))
      {
        return VK_ERROR_UNKNOWN;
      }

    std::map<std::string, const llama2::Variable *> variables;
    for (auto const &v : chkpts.variables ())
      {
        variables[v.name ()] = &v;
      }

    command_->begin ();
    // input layer
    {
      const auto *embeddings = variables["input/embeddings"];
      const auto *rms_norm_weight = variables["input/rms_norm/weight"];
      VkTensor vkembeddings (1, embeddings->shape (0), embeddings->shape (1),
                             gpu_);
      VkTensor vkrmsnorm_weight (1, 1, rms_norm_weight->shape (0), gpu_);

      if ((ret = vkembeddings.create ()) != VK_SUCCESS
          || (ret = vkrmsnorm_weight.create ()) != VK_SUCCESS)
        {
          return ret;
        }

      ret = command_->upload (embeddings->f32_values ().data (),
                              embeddings->f32_values_size (), vkembeddings);
      if (ret != VK_SUCCESS)
        {
          return ret;
        }
      ret = command_->upload (rms_norm_weight->f32_values ().data (),
                              rms_norm_weight->f32_values_size (),
                              vkrmsnorm_weight);
      if (ret != VK_SUCCESS)
        {
          return ret;
        }

      input_layer_
          = new InputLayer (gpu_, command_, vkembeddings, vkrmsnorm_weight);

      output_layer_ = new OutputLayer (gpu_, command_, vkembeddings);

      if ((ret = input_layer_->init ()) != VK_SUCCESS
          || (ret = output_layer_->init ()) != VK_SUCCESS)
        {
          return ret;
        }
    }

    // blocks
    {
      char vname[512];
      for (int b = 0; b < 6; ++b)
        {
          ::snprintf (vname, sizeof (vname), "block_%d/rms_norm_1/weight", b);
          const auto *rms_norm_weight_1 = variables[vname];
          ::snprintf (vname, sizeof (vname), "block_%d/rms_norm_2/weight", b);
          const auto *rms_norm_weight_2 = variables[vname];

          VkTensor vkrmsnorm_weight_1 (1, 1, rms_norm_weight_1->shape (0),
                                       gpu_);
          VkTensor vkrmsnorm_weight_2 (1, 1, rms_norm_weight_2->shape (0),
                                       gpu_);
          if ((ret = vkrmsnorm_weight_1.create ()) != VK_SUCCESS
              || (ret = vkrmsnorm_weight_2.create ()) != VK_SUCCESS)
            {
              return ret;
            }

          ret = command_->upload (rms_norm_weight_1->f32_values ().data (),
                                  rms_norm_weight_1->f32_values_size (),
                                  vkrmsnorm_weight_1);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          ret = command_->upload (rms_norm_weight_2->f32_values ().data (),
                                  rms_norm_weight_2->f32_values_size (),
                                  vkrmsnorm_weight_2);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          std::vector<VkTensor> Wk, Wq, Wv;
          for (int h = 0; h < 8; ++h)
            {
              ::snprintf (vname, sizeof (vname), "block_%d/Wk/head_%d", b, h);
              const auto *wk = variables[vname];
              ::snprintf (vname, sizeof (vname), "block_%d/Wq/head_%d", b, h);
              const auto *wq = variables[vname];
              ::snprintf (vname, sizeof (vname), "block_%d/Wv/head_%d", b, h);
              const auto *wv = variables[vname];

              VkTensor vkWk (1, wk->shape (0), wk->shape (1), gpu_);
              VkTensor vkWq (1, wq->shape (0), wq->shape (1), gpu_);
              VkTensor vkWv (1, wv->shape (0), wv->shape (1), gpu_);
              if ((ret = vkWk.create ()) != VK_SUCCESS
                  || (ret = vkWq.create ()) != VK_SUCCESS
                  || (ret = vkWv.create ()) != VK_SUCCESS)
                {
                  return ret;
                }

              ret = command_->upload (wk->f32_values ().data (),
                                      wk->f32_values_size (), vkWk);
              if (ret != VK_SUCCESS)
                {
                  return ret;
                }

              ret = command_->upload (wq->f32_values ().data (),
                                      wq->f32_values_size (), vkWq);
              if (ret != VK_SUCCESS)
                {
                  return ret;
                }

              ret = command_->upload (wv->f32_values ().data (),
                                      wv->f32_values_size (), vkWv);
              if (ret != VK_SUCCESS)
                {
                  return ret;
                }

              Wk.push_back (vkWk);
              Wq.push_back (vkWq);
              Wv.push_back (vkWv);
            }

          ::snprintf (vname, sizeof (vname), "block_%d/Wo", b);
          const auto *wo = variables[vname];

          VkTensor Wo (1, wo->shape (0), wo->shape (1), gpu_);
          if ((ret = Wo.create ()) != VK_SUCCESS)
            {
              return ret;
            }
          ret = command_->upload (wo->f32_values ().data (),
                                  wo->f32_values_size (), Wo);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          ::snprintf (vname, sizeof (vname), "block_%d/feed_forward/w1", b);
          const auto *w1 = variables[vname];
          ::snprintf (vname, sizeof (vname), "block_%d/feed_forward/w2", b);
          const auto *w2 = variables[vname];
          ::snprintf (vname, sizeof (vname), "block_%d/feed_forward/w3", b);
          const auto *w3 = variables[vname];

          VkTensor vkw1 (1, w1->shape (0), w1->shape (1), gpu_);
          VkTensor vkw2 (1, w2->shape (0), w2->shape (1), gpu_);
          VkTensor vkw3 (1, w3->shape (0), w3->shape (1), gpu_);
          if ((ret = vkw1.create ()) != VK_SUCCESS
              || (ret = vkw2.create ()) != VK_SUCCESS
              || (ret = vkw3.create ()) != VK_SUCCESS)
            {
              return ret;
            }

          ret = command_->upload (w1->f32_values ().data (),
                                  w1->f32_values_size (), vkw1);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          ret = command_->upload (w2->f32_values ().data (),
                                  w2->f32_values_size (), vkw2);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          ret = command_->upload (w3->f32_values ().data (),
                                  w3->f32_values_size (), vkw3);
          if (ret != VK_SUCCESS)
            {
              return ret;
            }

          Llama2Block::RmsNormParams rmsnorm_params
              = { vkrmsnorm_weight_1, vkrmsnorm_weight_2 };
          Llama2Block::TransformerParams transformer_params
              = { Wk, Wq, Wv, Wo, 128, 512 };
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
    command_->submit_and_wait ();
    return VK_SUCCESS;
  }

  std::vector<uint32_t>
  operator() (std::vector<uint32_t> const &toks)
  {
    auto t1 = std::chrono::high_resolution_clock::now ();
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
    std::vector<VkTensor> tmps;
    tmps.push_back (X);

    for (int i = 0; i < blocks_.size (); ++i)
      {
        auto *block = blocks_[i];
        X = (*block) (X);
        tmps.push_back (X);
      }

    VkTensor output = (*output_layer_) (X);
    std::vector<uint32_t> buf (output.size ());

    ret = command_->download (output, buf.data (), buf.size ());
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at downloadding outputs");
      }

    ret = command_->end ();
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at end commands");
      }

    auto t2 = std::chrono::high_resolution_clock::now ();
    ret = command_->submit_and_wait ();
    if (ret != VK_SUCCESS)
      {
        throw std::runtime_error ("failed at submit");
      }

    auto t3 = std::chrono::high_resolution_clock::now ();
    auto record_cost
        = std::chrono::duration_cast<std::chrono::microseconds> (t2 - t1)
              .count ();
    auto sharder_cost
        = std::chrono::duration_cast<std::chrono::microseconds> (t3 - t2)
              .count ();
    fprintf (stderr, "record cost: %ldus, shader cost: %ldus\n", record_cost,
             sharder_cost);
    return buf;
  }

private:
  GPUDevice *gpu_;
  Command *command_;
  InputLayer *input_layer_;
  OutputLayer *output_layer_;
  std::vector<Llama2Block *> blocks_;
};

#endif
