#pragma once

#include "Utility.h"
#include "zensim/math/Vec.h"
#include "zensim/tpls/fmt/core.h"

namespace zs {

  /// ref: http://docs.ros.org/en/kinetic/api/gtsam/html/SelfAdjointEigenSolver_8h_source.html
  template <typename VecTM,
            enable_if_all<std::is_floating_point_v<typename VecTM::value_type>, VecTM::dim == 2,
                          VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
                          VecTM::template range_t<0>::value == 2> = 0>
  constexpr auto eigen_decomposition(VecInterface<VecTM> &M) noexcept {
    using T = typename VecTM::value_type;
    using MatT = typename VecTM::template variant_vec<T, typename VecTM::extents>;
    using VecT =
        typename VecTM::template variant_vec<T, integer_seq<typename VecTM::index_type, 2>>;
    MatT eivecs{};
    VecT eivals{};
    T shift = trace(M) / (T)2;
    auto scaledMat = M.clone();
    scaledMat(0, 1) = M(1, 0);
    scaledMat(0, 0) -= shift;
    scaledMat(1, 1) -= shift;
    T scale = scaledMat.abs().max();
    if (scale > (T)0) scaledMat /= scale;
    // compute eigenvalues
    {
      T t0 = (T)0.5
             * zs::sqrt(zs::sqr(scaledMat(0, 0) - scaledMat(1, 1))
                        + (T)4 * zs::sqr(scaledMat(1, 0)));
      T t1 = (T)0.5 * (scaledMat(0, 0) + scaledMat(1, 1));
      eivals(0) = t1 - t0;
      eivals(1) = t1 + t0;
    }
    // compute eigenvectors
    {
      if (eivals(1) - eivals(0) <= zs::abs(eivals(1)) * limits<T>::epsilon())
        eivecs = MatT::identity();
      else {
        for (int d = 0; d != 2; ++d) scaledMat(d, d) -= eivals(1);
        const T a2 = zs::sqr(scaledMat(0, 0));
        const T c2 = zs::sqr(scaledMat(1, 1));
        const T b2 = zs::sqr(scaledMat(1, 0));
        if (a2 > c2) {
          auto coeff = zs::sqrt(a2 + b2);
          eivecs(0, 1) = -scaledMat(1, 0) / coeff;
          eivecs(1, 1) = scaledMat(0, 0) / coeff;
        } else {
          auto coeff = zs::sqrt(c2 + b2);
          eivecs(0, 1) = -scaledMat(1, 1) / coeff;
          eivecs(1, 1) = scaledMat(1, 0) / coeff;
        }
        auto tmp = col(eivecs, 1).orthogonal();
        eivecs(0, 0) = tmp(0);
        eivecs(1, 0) = tmp(1);
      }
    }
    eivals *= scale;
    eivals += shift;
    return zs::make_tuple(eivals, eivecs);
  }

  template <typename VecTM,
            enable_if_all<std::is_floating_point_v<typename VecTM::value_type>, VecTM::dim == 2,
                          VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
                          VecTM::template range_t<0>::value == 3> = 0>
  constexpr auto eigen_decomposition(VecInterface<VecTM> &M) noexcept {
    using T = typename VecTM::value_type;
    using MatT = typename VecTM::template variant_vec<T, typename VecTM::extents>;
  }

}  // namespace zs