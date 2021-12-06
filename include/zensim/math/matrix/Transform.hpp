#pragma once
#include "QRSVD.hpp"

namespace zs::math {

template <typename VecTM,
            enable_if_all<VecTM::dim == 2, (VecTM::template range<0> == VecTM::template range<1>) > = 0>
  constexpr auto decompose_transform(const VecInterface<VecTM>& m,
                                     bool applyOnColumn = true) noexcept {
    constexpr auto dim = VecTM::template range<0> - 1;
  static_assert(VecTM::template range<0> <= 4 && (VecTM::template range<0> > 1),
                "transformation should be of 2x2, 3x3 or 4x4 shape only.");
    using Mat = decltype(m.clone());
    using ValT = typename VecTM::value_type;
    using Tn = typename VecTM::index_type;
    auto H = m.clone();
    if (!applyOnColumn) H = H.transpose();
    // T
    auto T = Mat::identity();
    for (Tn i = 0; i != dim; ++i) {
      T(i, dim) = H(i, dim);
      H(i, dim) = 0;
    }
    // RS
    typename VecTM::template variant_vec<ValT, integer_seq<Tn, dim, dim>> L{};
    for (Tn i = 0; i != dim; ++i)
      for (Tn j = 0; j != dim; ++j) L(i, j) = H(i, j);
    auto [R_, S_] = polar_decomposition(L);
    auto R{Mat::zeros()}, S{Mat::zeros()};
    for (Tn i = 0; i != dim; ++i)
      for (Tn j = 0; j != dim; ++j) {
        R(i, j) = R_(i, j);
        S(i, j) = S_(i, j);
      }
    R(dim, dim) = S(dim, dim) = (ValT)1;
    return std::make_tuple(T, R, S);
  }

}