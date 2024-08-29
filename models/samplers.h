#ifndef __VKLLAMA_SAMPLERS_H__
#define __VKLLAMA_SAMPLERS_H__
#include <random>
#include <utility>
#include <vector>

class Sampler
{
public:
  virtual int sample (const float *logits, size_t n) = 0;
  Sampler ();
  virtual ~Sampler () = default;

protected:
  virtual size_t sample_from_prob (const float *probs, size_t n);
  void softmax (std::vector<std::pair<float, int> > &);

private:
  std::random_device rd_;
  std::mt19937 g_;
};

class TopkSampler : public Sampler
{
public:
  TopkSampler (size_t topk = 40);
  int sample (const float *logits, size_t n) override;

private:
  size_t topk_;
};
#endif
