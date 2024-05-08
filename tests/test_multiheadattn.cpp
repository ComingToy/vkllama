#include "Eigen/Eigen"
#include "core/command.h"
#include "core/gpu_device.h"
#include "ops/multiheadattention.h"
#include "test_common.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "gtest/gtest.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <math.h>
#include <utility>
#include <vector>

namespace
{
struct TestMultiheadattnParams
{
  const int C;
  const int H;
  const int W;
  const int HDIM;
  const int HEADS;
  const int MAXLEN;
};

static std::pair<Tensor<3>, Tensor<3> >
precompute_freq (const int maxlen, const int dim)
{
  std::vector<float> freq;
  std::generate_n (
      std::back_inserter (freq), dim / 2, [n = 0, d = (float)dim] () mutable {
        float f = 1.0f / std::pow (10000.0f, static_cast<float> (n) / d);
        n += 2;
        return f;
      });

  std::vector<float> freqc (maxlen * dim / 2), freqs (maxlen * dim / 2);
  for (int i = 0; i < maxlen; ++i)
    {
      for (int k = 0; k < dim / 2; ++k)
        {
          auto f = freq[k] * static_cast<float> (i);
          freqc[i * dim / 2 + k] = std::cos (f);
          freqs[i * dim / 2 + k] = std::sin (f);
        }
    }

  Tensor<3> freqc_tensor = TensorMap<3> (freqc.data (), 1, maxlen, dim / 2);
  Tensor<3> freqs_tensor = TensorMap<3> (freqs.data (), 1, maxlen, dim / 2);

  return { freqc_tensor, freqs_tensor };
}

static std::pair<Tensor<3>, Tensor<3> >
apply_rope (Tensor<3> const &query, Tensor<3> const &key,
            Tensor<3> const &freqc, Tensor<3> const &freqs)
{
  auto _apply_rope
      = [] (Tensor<3> input_x, Tensor<3> input_freqc, Tensor<3> input_freqs)

  {
    // apply rope to query
    Eigen::array<Eigen::Index, 4> dims
        = { input_x.dimension (0), input_x.dimension (1),
            input_x.dimension (2) / 2, (Eigen::Index)2 };
    Tensor<4> query_host = input_x.reshape (dims);
    Tensor<3> query_host_r = query_host.chip<3> (0);
    Tensor<3> query_host_i = query_host.chip<3> (1);

    Tensor<3> query_host_or (query_host_r.dimensions ());
    Tensor<3> query_host_oi (query_host_r.dimensions ());
    for (int i = 0; i < query_host_or.dimension (0); ++i)
      {
        query_host_or.chip<0> (i)
            = query_host_r.chip<0> (i) * input_freqc.chip<0> (0)
              - query_host_i.chip<0> (i) * input_freqs.chip<0> (0);
        query_host_oi.chip<0> (i)
            = query_host_r.chip<0> (i) * input_freqs.chip<0> (0)
              + query_host_i.chip<0> (i) * input_freqc.chip<0> (0);
      }

    // auto output_query_host_ref =
    // query_host_or.reshape(Eigen::array<Eigen::Index,
    // 4>{query_host_or.dimension(0), query_host_or.dimension(1),
    // query_host_or.dimension(2),
    // (Eigen::Index)1}).concatenate(query_host_oi.reshape(Eigen::array<Eigen::Index,
    // 4>{query_host_oi.dimension(0), query_host_oi.dimension(1),
    // query_host_oi.dimension(2), (Eigen::Index)1}), 3);
    Eigen::array<Eigen::Index, 4> out_dims
        = { query_host_or.dimension (0), query_host_or.dimension (1),
            query_host_or.dimension (2), (Eigen::Index)1 };
    Tensor<4> reshaped_query_host_or = query_host_or.reshape (out_dims);
    Tensor<4> reshaped_query_host_oi = query_host_oi.reshape (out_dims);
    Tensor<4> output
        = reshaped_query_host_or.concatenate (reshaped_query_host_oi, 3);

    Tensor<3> output_ = output.reshape (input_x.dimensions ());
    return output_;
  };

  auto rotated_query = _apply_rope (query, freqc, freqs);
  auto rotated_key = _apply_rope (key, freqc, freqs);

  return { rotated_query, rotated_key };
}

class TestMultiheadattn
    : public ::testing::TestWithParam<TestMultiheadattnParams>
{
public:
  GPUDevice *gpu_;
  Command *command_;
  void
  SetUp () override
  {
    gpu_ = new GPUDevice ();
    command_ = new Command (gpu_);
    ASSERT_EQ (gpu_->init (), VK_SUCCESS);
    ASSERT_EQ (command_->init (), VK_SUCCESS);
  }

  void
  TearDown () override
  {
    delete command_;
    delete gpu_;
  }
};

TEST_P (TestMultiheadattn, test_multiheadattn)
{
  auto params = GetParam ();
  ASSERT_EQ (command_->begin (), VK_SUCCESS) << "failed at begin commands";
  auto input0 = random_tensor<float> (gpu_, command_, params.C, params.H, params.W);
  ASSERT_TRUE (input0) << "failed at create tensor";

  std::vector<VkTensor> wk, wq, wv;
  std::vector<std::vector<float> > wk_bufs, wq_bufs, wv_bufs;
  VkTensor wo;
  std::vector<float> wo_bufs;

  {
    auto pwo
        = random_tensor<float> (gpu_, command_, params.C, params.HDIM * params.HEADS,
                         input0->first.width ());
    ASSERT_TRUE (pwo) << "failed at create tensor";
    wo = pwo->first;
    wo_bufs.swap (pwo->second);

    for (int i = 0; i < params.HEADS; ++i)
      {
        auto k
            = random_tensor<float> (gpu_, command_, params.C, params.W, params.HDIM);
        auto q
            = random_tensor<float> (gpu_, command_, params.C, params.W, params.HDIM);
        auto v
            = random_tensor<float> (gpu_, command_, params.C, params.W, params.HDIM);

        ASSERT_TRUE (k && q && v) << "failed at create tensor";
        wk.push_back (k->first);
        wq.push_back (q->first);
        wv.push_back (v->first);
        wk_bufs.push_back (std::move (k->second));
        wq_bufs.push_back (std::move (q->second));
        wv_bufs.push_back (std::move (v->second));
      }
  }

  MultiHeadAttention attn_op (gpu_, command_, wk, wq, wv, wo, params.MAXLEN,
                              params.HDIM);
  ASSERT_EQ (attn_op.init (), VK_SUCCESS) << "failed at init attention op";
  VkTensor output;
  ASSERT_EQ (attn_op (input0->first, output), VK_SUCCESS)
      << "failed at infer attn op";
  std::vector<float> output_buf (output.size ());
  ASSERT_EQ (
      command_->download (output, output_buf.data (), output_buf.size ()),
      VK_SUCCESS)
      << "failed at download output";

  ASSERT_EQ (command_->end (), VK_SUCCESS) << "failed at end commands";
  ASSERT_EQ (command_->submit_and_wait (), VK_SUCCESS)
      << "failed at submit commands";

  Tensor<3> vk_output_tensor
      = TensorMap<3> (output_buf.data (), output.channels (), output.height (),
                      output.width ());
  // std::cerr << "mean of vk_output_tensor: " << vk_output_tensor.mean ()
  //           << std::endl;

  // std::vector<Tensor<3> > K (wk.size ()), Q (wq.size ()), V (wv.size ());
  Tensor<3> input_tensor
      = TensorMap<3> (input0->second.data (), input0->first.channels (),
                      input0->first.height (), input0->first.width ());

  auto [freqc, freqs] = precompute_freq (input0->first.height (), params.HDIM);
  Tensor<3> eigen_output_tensor;
  {
    std::vector<Tensor<3> > heads;
    for (int i = 0; i < params.HEADS; ++i)
      {
        auto k = TensorMap<3> (wk_bufs[i].data (), wk[i].channels (),
                               wk[i].height (), wk[i].width ());

        auto q = TensorMap<3> (wq_bufs[i].data (), wq[i].channels (),
                               wq[i].height (), wq[i].width ());

        auto v = TensorMap<3> (wv_bufs[i].data (), wv[i].channels (),
                               wv[i].height (), wv[i].width ());

        auto K = eigen_tensor_matmul (input_tensor, k, 0);
        auto Q = eigen_tensor_matmul (input_tensor, q, 0);
        auto V = eigen_tensor_matmul (input_tensor, v, 0);

        auto [rotated_q, rotated_k] = apply_rope (Q, K, freqc, freqs);

        Eigen::array<Eigen::IndexPair<int>, 1> dims
            = { Eigen::IndexPair<int> (1, 0) };

        Eigen::array<Eigen::Index, 3> trans = { 0, 2, 1 };
        Tensor<3> rotated_transpoed_k = rotated_k.shuffle (trans);

        float scale = 1.0f / std::sqrt (static_cast<float> (Q.dimension (2)));

        Tensor<3> attn_scores (V.dimension (0), V.dimension (1),
                               V.dimension (1));

        for (int c = 0; c < V.dimension (0); ++c)
          {
            attn_scores.chip<0> (c)
                = rotated_q.chip<0> (c).contract (
                      rotated_transpoed_k.chip<0> (c), dims)
                  * scale;
          }

        Tensor<3> normalized_attn_scores (attn_scores.dimensions ());
        {
          Tensor<3> exps;
          Tensor<3> m;
          Eigen::array<Eigen::Index, 1> max_dims = { 2 };
          Eigen::array<Eigen::Index, 3> bias_dims
              = { attn_scores.dimension (0), attn_scores.dimension (1), 1 };
          Eigen::array<Eigen::Index, 3> broadcasts
              = { 1, 1, attn_scores.dimension (2) };
          auto debias = attn_scores
                        - attn_scores.maximum (max_dims)
                              .reshape (bias_dims)
                              .broadcast (broadcasts);
          exps = debias.exp ();
          m = exps.sum (max_dims).reshape (bias_dims).broadcast (broadcasts);
          normalized_attn_scores = exps / m;
        }

        Tensor<3> head (V.dimensions ());
        {
          for (int c = 0; c < normalized_attn_scores.dimension (0); ++c)
            {
              head.chip<0> (c) = normalized_attn_scores.chip<0> (c).contract (
                  V.chip<0> (c), dims);
            }
        }
        heads.push_back (head);
#if 0
        std::cerr << "head " << i << ", mean of K = " << K.mean ()
                  << ", Q = " << Q.mean () << ", V = " << V.mean ()
                  << ", Kr = " << rotated_k.mean ()
                  << ", Qr = " << rotated_q.mean ()
                  << ", transpoed_k = " << rotated_transpoed_k.mean ()
                  << ", scale = " << scale
                  << ", attn_scores = " << attn_scores.mean ()
                  << ", normalized_attn_scores = "
                  << normalized_attn_scores.mean ()
                  << ", head = " << head.mean () << std::endl;
#endif
      }

    Tensor<3> tmp = heads[0];
    Tensor<3> concated_head;
    for (int i = 1; i < heads.size (); ++i)
      {
        concated_head = tmp.concatenate (heads[i], 2);
        tmp = concated_head;
      }

    Tensor<3> wo_tensor = TensorMap<3> (wo_bufs.data (), wo.channels (),
                                        wo.height (), wo.width ());

    // std::cerr << "mean of concated_head = " << concated_head.mean ()
    //           << ", wo mean: " << wo_tensor.mean () << std::endl;
    eigen_output_tensor = eigen_tensor_matmul (concated_head, wo_tensor, 0);
  }

  Tensor<0> m0 = vk_output_tensor.mean ();
  Tensor<0> m1 = eigen_output_tensor.mean ();

  Tensor<3> err (vk_output_tensor.dimensions ());
  err.setConstant (1e-3);
  Tensor<0> diff = ((vk_output_tensor - eigen_output_tensor).abs () > err)
                       .cast<float> ()
                       .sum ();

  ASSERT_FLOAT_EQ (*diff.data (), .0f);
  // std::cerr << "mean of vk_output_tensor: " << vk_output_tensor.mean ()
  //           << std::endl
  //           << "mean of eigen_output_tensor: " << eigen_output_tensor.mean
  //           ()
  //           << std::endl;
}

std::vector<TestMultiheadattnParams> params
    = { { 1, 256, 64, 512, 8, 512 }, { 3, 253, 64, 512, 3, 321 } };
INSTANTIATE_TEST_SUITE_P (test_multiheadattn, TestMultiheadattn,
                          ::testing::ValuesIn (params));
}
