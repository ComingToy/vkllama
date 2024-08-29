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
  std::sort (
      logits.begin (), logits.end (),
      [] (auto const &lhs, auto const &rhs) { return lhs.first > rhs.first; });

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

  softmax (tmp);
  std::vector<float> probs;
  std::transform (tmp.cbegin (), tmp.cbegin () + std::min (topk_, tmp.size ()),
                  std::back_inserter (probs),
                  [] (auto const &v) { return v.first; });

  auto i = sample_from_prob (probs.data (), probs.size ());

  return tmp[i].second;
}
