#pragma once
#include "QRSVD.hpp"
#include "zensim/math/Rotation.hpp"

namespace zs::math {

  /// ref: openvdb/math/Mat4.h
  template <typename T_, int dim_> struct Transform : zs::vec<T_, dim_ + 1, dim_ + 1> {
    using value_type = T_;
    static constexpr int dim = dim_;

    using mat_type = zs::vec<value_type, dim + 1, dim + 1>;

    constexpr decltype(auto) self() const { return static_cast<const mat_type &>(*this); }
    constexpr decltype(auto) self() { return static_cast<mat_type &>(*this); }

    /// translation
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr void setToTranslation(const VecInterface<VecT> &v) noexcept {
      self() = mat_type::identity();
      for (int i = 0; i != dim_; ++i) self()(dim_, i) = v[i];
    }

    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr void preTranslate(const VecInterface<VecT> &v) noexcept {
      Transform tr{};
      tr.setToTranslation(v);
      self() = tr * self();
    }

    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr void postTranslate(const VecInterface<VecT> &v) noexcept {
      Transform tr{};
      tr.setToTranslation(v);
      self() = self() * tr;
    }

    /// scale
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr void setToScale(const VecInterface<VecT> &v) noexcept {
      self() = mat_type::identity();
      for (int d = 0; d != dim_; ++d) self()(d, d) = v[d];
    }

    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr void preScale(const VecInterface<VecT> &v) noexcept {
      for (int i = 0; i != dim_; ++i)
        for (int j = 0; j != dim_ + 1; ++j) self()(i, j) *= v[i];
    }

    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr void postScale(const VecInterface<VecT> &v) noexcept {
      for (int i = 0; i != dim_ + 1; ++i)
        for (int j = 0; j != dim_; ++j) self()(i, j) *= v[j];
    }

    /// rotation
    template <typename T, enable_if_t<is_convertible_v<T, value_type>> = 0>
    void setToRotation(const Rotation<T, dim> &r) noexcept {
      self() = mat_type::zeros();
      for (int i = 0; i != dim_; ++i)
        for (int j = 0; j != dim_; ++j) self()(i, j) = r(j, i);
      self()(dim_, dim_) = 1;
    }

    template <typename T, enable_if_t<is_convertible_v<T, value_type>> = 0>
    void preRotate(const Rotation<T, dim> &v) noexcept {
      Transform rot{};
      rot.setToRotation(v);
      self() = rot * self();
    }

    template <typename T, enable_if_t<is_convertible_v<T, value_type>> = 0>
    void postRotate(const Rotation<T, dim> &v) noexcept {
      Transform rot{};
      rot.setToRotation(v);
      self() = self() * rot;
    }
  };

  template <typename VecTM, enable_if_all<VecTM::dim == 2, VecTM::template range_t<0>::value
                                                               == VecTM::template range_t<1>::value>
                            = 0>
  constexpr auto decompose_transform(const VecInterface<VecTM> &m,
                                     bool applyOnColumn = true) noexcept {
    constexpr auto dim = VecTM::template range_t<0>::value - 1;
    static_assert(VecTM::template range_t<0>::value <= 4 && (VecTM::template range_t<0>::value > 1),
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
    typename VecTM::template variant_vec<ValT, integer_sequence<Tn, dim, dim>> L{};
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
    if (applyOnColumn) return zs::make_tuple(T, R, S);
    return zs::make_tuple(S.transpose(), R.transpose(), T.transpose());
  }

  template <
      typename VecTM, typename VecTS, typename VecTR, typename VecTT,
      enable_if_all<VecTM::dim == 2,
                    VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
                    VecTS::dim == 2,
                    VecTS::template range_t<0>::value + 1 == VecTM::template range_t<0>::value,
                    VecTS::template range_t<1>::value + 1 == VecTM::template range_t<0>::value,
                    VecTR::dim == 2,
                    VecTR::template range_t<0>::value + 1 == VecTM::template range_t<0>::value,
                    VecTR::template range_t<1>::value + 1 == VecTM::template range_t<0>::value,
                    VecTT::dim == 1,
                    VecTT::template range_t<0>::value + 1 == VecTM::template range_t<0>::value,
                    is_floating_point_v<typename VecTM::value_type>,
                    is_floating_point_v<typename VecTS::value_type>,
                    is_floating_point_v<typename VecTR::value_type>,
                    is_floating_point_v<typename VecTT::value_type>>
      = 0>
  constexpr void decompose_transform(const VecInterface<VecTM> &m, VecInterface<VecTS> &s,
                                     VecInterface<VecTR> &r, VecInterface<VecTT> &t,
                                     bool applyOnColumn = true) noexcept {
    constexpr auto dim = VecTM::template range_t<0>::value - 1;
    static_assert(VecTM::template range_t<0>::value <= 4 && (VecTM::template range_t<0>::value > 1),
                  "transformation should be of 2x2, 3x3 or 4x4 shape only.");
    using ValT = typename VecTM::value_type;
    using Tn = typename VecTM::index_type;
    auto H = m.clone();
    if (!applyOnColumn) H = H.transpose();
    // T
    for (Tn i = 0; i != dim; ++i) t(i) = H(i, dim);
    // RS
    typename VecTM::template variant_vec<ValT, integer_sequence<Tn, dim, dim>> L{};
    for (Tn i = 0; i != dim; ++i)
      for (Tn j = 0; j != dim; ++j) L(i, j) = H(i, j);
    auto [R_, S_] = polar_decomposition(L);

    for (Tn i = 0; i != dim; ++i)
      for (Tn j = 0; j != dim; ++j) {
        r(i, j) = R_(i, j);
        s(i, j) = S_(i, j);
      }
  }

}  // namespace zs::math