#ifndef __VKLLAMA_TEST_COMMON_H__
#define __VKLLAMA_TEST_COMMON_H__
#include <math.h>

#include <memory>
#include <optional>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/command.h"
#include "core/tensor.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "gtest/gtest.h"

template <typename T>
inline void
random_vec (T *v, const int n, const T min, const T max)
{
  std::random_device rd;
  std::mt19937 e2 (rd ());

  if constexpr (std::is_same<typename std::remove_const<T>::type,
                             __vkllama_fp16_t>::value)
    {
      float fmin = Eigen::half (min.u16);
      float fmax = Eigen::half (max.u16);
      std::uniform_real_distribution<float> dist (fmin, fmax);

      static_assert (sizeof (__vkllama_fp16_t) == sizeof (Eigen::half::x),
                     "Eigen::half is not 16bit storage.");
      for (int i = 0; i < n; ++i)
        {
          auto v16 = Eigen::half (dist (e2)).x;

          __vkllama_fp16_t f = { .u16 = *reinterpret_cast<uint16_t *> (&v16) };
          v[i] = f;
        }

      return;
    }

  if constexpr (std::is_same<typename std::remove_const<T>::type,
                             Eigen::half>::value)
    {
      float fmin = Eigen::half (min);
      float fmax = Eigen::half (max);
      std::uniform_real_distribution<float> dist (fmin, fmax);

      for (int i = 0; i < n; ++i)
        {
          v[i] = Eigen::half (dist (e2));
        }
    }
  else
    {
      typename std::conditional<
          std::is_floating_point<T>::value, std::uniform_real_distribution<T>,
          std::uniform_int_distribution<T> >::type dist (min, max);

      for (int i = 0; i < n; ++i)
        {
          v[i] = dist (e2);
        }
    }
}

template <typename T>
inline T
random_number (T min, T max)
{
  std::random_device rd;
  std::mt19937 e2 (rd ());
  std::uniform_real_distribution<T> dist (min, max);
  return dist (e2);
}

template <typename T>
inline absl::optional<std::pair<vkllama::Tensor, std::vector<T> > >
random_tensor (vkllama::GPUDevice *dev, vkllama::Command *command, const int c,
               const int h, const int w, const T min = T (-1),
               const T max = T (1))
{

  using tensor_dtype_t =
      typename std::conditional<std::is_same<Eigen::half, T>::value,
                                __vkllama_fp16_t, T>::type;

  vkllama::Tensor tensor (c, h, w, dev,
                          vkllama::Tensor::to_dtype<tensor_dtype_t> ());
  if (tensor.create () != absl::OkStatus ())
    {
      return {};
    }

  const int n = c * h * w;
  std::vector<T> buf (n);

  random_vec (buf.data (), n, min, max);

  auto ret = command->upload ((tensor_dtype_t *)buf.data (), n, tensor);
  if (ret != absl::OkStatus ())
    {
      return {};
    }

  return std::pair<vkllama::Tensor, std::vector<T> > (tensor, buf);
}

template <int NumIndices_ = 3>
using TensorMap
    = Eigen::TensorMap<Eigen::Tensor<float, NumIndices_, Eigen::RowMajor> >;

template <typename Scalar, int NumIndices_ = 3>
using _TensorMap
    = Eigen::TensorMap<Eigen::Tensor<Scalar, NumIndices_, Eigen::RowMajor> >;

template <typename Scalar, int NumIndices_>
using _Tensor = Eigen::Tensor<Scalar, NumIndices_, Eigen::RowMajor>;

inline _Tensor<float, 3>
eigen_tensor_matmul (_Tensor<float, 3> lhs, _Tensor<float, 3> rhs_,
                     const int broadcast_type, const bool transpose_b = false)
{

  _Tensor<float, 3> rhs;

  if (transpose_b)
    {
      Eigen::array<Eigen::Index, 3> trans = { 0, 2, 1 };
      rhs = rhs_.shuffle (trans);
    }
  else
    {
      rhs = rhs_;
    }

  _Tensor<float, 3> eigen_output (lhs.dimension (0), lhs.dimension (1),
                                  rhs.dimension (2));
  Eigen::array<Eigen::IndexPair<int>, 1> dims
      = { Eigen::IndexPair<int> (1, 0) };
  if (broadcast_type == 0)
    {
      for (int i = 0; i < lhs.dimension (0); ++i)
        {
          eigen_output.chip<0> (i)
              = lhs.chip<0> (i).contract (rhs.chip<0> (i), dims);
        }
    }
  else if (broadcast_type == 1)
    {
      for (int i = 0; i < lhs.dimension (0); ++i)
        {
          eigen_output.chip<0> (i)
              = lhs.chip<0> (i).contract (rhs.chip<0> (0), dims);
        }
    }
  else if (broadcast_type == 2)
    {
      for (int i = 0; i < lhs.dimension (0); ++i)
        {
          eigen_output.chip<0> (i)
              = lhs.chip<0> (0).contract (rhs.chip<0> (i), dims);
        }
    }

  return eigen_output;
}

template <typename _Params>
class VkllamaTestWithParam : public ::testing::TestWithParam<_Params>
{
public:
  void
  SetUp () override
  {
    gpu_ = new vkllama::GPUDevice ();
    command_ = new vkllama::Command (gpu_);

    gpu_->init ();
    command_->init ();
  }

  void
  TearDown () override
  {
    delete command_;
    delete gpu_;
  }

  vkllama::GPUDevice *gpu_;
  vkllama::Command *command_;
};
#endif
