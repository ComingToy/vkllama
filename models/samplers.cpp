#include "samplers.h"
#include <algorithm>
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
  std::vector<std::pair<float, int> > tmp (n);
  for (size_t i = 0; i < n; ++i)
    {
      tmp[i] = std::make_pair (logits[i], (int)i);
    }

  std::sort (tmp.begin (), tmp.end (), [] (auto const &lhs, auto const &rhs) {
    return lhs.first > rhs.first;
  });

  tmp.resize (topk_);

  softmax (tmp);

  std::vector<float> probs;
  probs.reserve (topk_);

  std::transform (tmp.cbegin (), tmp.cbegin () + std::min (topk_, tmp.size ()),
                  std::back_inserter (probs),
                  [] (auto const &v) { return v.first; });

  auto i = sample_from_prob (probs.data (), probs.size ());

  return tmp[i].second;
}

TopPSampler::TopPSampler (const float p) : p_ (p) {}
int
TopPSampler::sample (const float *logits, size_t n)
{
  std::vector<std::pair<float, int> > tmp;
  for (size_t i = 0; i < n; ++i)
    {
      tmp.push_back (std::make_pair (logits[i], (int)i));
    }

  std::sort (tmp.begin (), tmp.end (), [] (auto const &lhs, auto const &rhs) {
    return lhs.first > rhs.first;
  });

  std::vector<std::pair<float, int> > top_p;
  {
    std::vector<std::pair<float, int> > probs (tmp);
    softmax (probs);

    float acc = .0f;
    for (size_t i = 0; i < tmp.size () && acc < p_; ++i)
      {
        top_p.push_back (tmp[i]);
        acc += probs[i].first;
      }
  }
  softmax (top_p);

  std::vector<float> probs;
  std::transform (top_p.cbegin (), top_p.cend (), std::back_inserter (probs),
                  [] (auto const &v) { return v.first; });

  auto i = sample_from_prob (probs.data (), probs.size ());

  return tmp[i].second;
}

