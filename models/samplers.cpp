#include "samplers.h"
#include <iterator>
#include <utility>
#include <vector>

Sampler::Sampler () : rd_ (), g_ (rd_ ()) {}

size_t
Sampler::sample_from_prob (const float *probs, size_t n)
{
  std::discrete_distribution<> dis (probs, probs + n);
  return dis (g_);
}

void
Sampler::softmax (std::vector<std::pair<float, int> > &logits)
{

  auto m = logits[0].first;

  float acc = .0f;
  for (size_t i = 0; i < logits.size (); ++i)
    {
      logits[i].first = std::exp (logits[i].first - m);
      acc += logits[i].first;
    }

  for (size_t i = 0; i < logits.size (); ++i)
    {
      logits[i].first /= acc;
    }
};

TopkSampler::TopkSampler (size_t topk) : topk_ (topk) {}

int
TopkSampler::sample (const float *logits, size_t n)
{
  std::vector<std::pair<float, int> > tmp;
  for (size_t i = 0; i < n; ++i)
    {
      tmp.push_back (std::make_pair (logits[i], (int)i));
    }

  std::sort (tmp.begin (), tmp.end (), [] (auto const &lhs, auto const &rhs) {
    return lhs.first > rhs.first;
  });

  std::vector<std::pair<float, int> > topk;
  for (size_t i = 0; i < topk_; ++i)
    {
      topk.push_back (tmp[i]);
    }

  softmax (topk);

  std::vector<float> probs;
  std::transform (
      topk.cbegin (), topk.cbegin () + std::min (topk_, topk.size ()),
      std::back_inserter (probs), [] (auto const &v) { return v.first; });

  auto i = sample_from_prob (probs.data (), probs.size ());

  return tmp[i].second;
}
