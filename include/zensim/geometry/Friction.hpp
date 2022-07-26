#pragma once

#include "zensim/math/MathUtils.h"
#include "zensim/math/Vec.h"
#include "zensim/math/VecInterface.hpp"

namespace zs {

  /// ref: ipc-sim/Codim-IPC, FEM/FRICTION_UTILS.h

  /// PP
  template <typename VecTA, typename VecTB,
            enable_if_all<VecTA::dim == 1, VecTA::extent == 3,
                          is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
  constexpr auto point_point_tangent_basis(const VecInterface<VecTA>& p0,
                                           const VecInterface<VecTB>& p1) {
    using T = typename VecTA::value_type;
    using Ti = typename VecTA::index_type;
    using RetT = typename VecTA::template variant_vec<T, integer_seq<Ti, 3, 2>>;
    RetT basis{};
    auto v01 = (p1 - p0);
    auto xCross = VecTA::init([](int i) -> T { return i == 0 ? 1 : 0; }).cross(v01);
    auto yCross = VecTA::init([](int i) -> T { return i == 1 ? 1 : 0; }).cross(v01);
    RM_CVREF_T(v01) c0{}, c1{};
    if (xCross.l2NormSqr() > yCross.l2NormSqr()) {
      auto c0 = xCross.normalized();
      auto c1 = v01.cross(xCross).normalized();
    } else {
      auto c0 = yCross.normalized();
      auto c1 = v01.cross(yCross).normalized();
    }
    for (int d = 0; d != 3; ++d) {
      basis(d, 0) = c0[d];
      basis(d, 1) = c1[d];
    }
    return basis;
  }

  /// PE
  template <typename VecTA, typename VecTB,
            enable_if_all<VecTA::dim == 1, VecTA::extent == 3,
                          is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
  constexpr auto point_edge_closest_point(const VecInterface<VecTA>& v0,
                                          const VecInterface<VecTB>& v1,
                                          const VecInterface<VecTB>& v2) {
    auto e12 = v2 - v1;
    return (v0 - v1).dot(e12) / e12.l2NormSqr();  // yita
  }
  template <typename VecTA, typename VecTB,
            enable_if_all<VecTA::dim == 1, VecTA::extent == 3,
                          is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
  constexpr auto point_edge_tangent_basis(const VecInterface<VecTA>& v0,
                                          const VecInterface<VecTB>& v1,
                                          const VecInterface<VecTB>& v2) {
    using T = typename VecTA::value_type;
    using Ti = typename VecTA::index_type;
    using RetT = typename VecTA::template variant_vec<T, integer_seq<Ti, 3, 2>>;
    RetT basis{};
    auto v12 = v2 - v1;
    auto c0 = v12.normalized();
    auto c1 = v12.cross(v0 - v1).normalized();
    for (int d = 0; d != 3; ++d) {
      basis(d, 0) = c0[d];
      basis(d, 1) = c1[d];
    }
    return basis;
  }

  /// PT
  template <typename VecTA, typename VecTB,
            enable_if_all<VecTA::dim == 1, VecTA::extent == 3,
                          is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
  constexpr auto point_triangle_closest_point(const VecInterface<VecTA>& v0,
                                              const VecInterface<VecTB>& v1,
                                              const VecInterface<VecTB>& v2,
                                              const VecInterface<VecTB>& v3) {
    using T = typename VecTA::value_type;
    using Ti = typename VecTA::index_type;
    using MT = typename VecTA::template variant_vec<T, integer_seq<Ti, 2, 3>>;
    MT basis{};
    auto r0 = v2 - v1;
    auto r1 = v3 - v1;
    for (int d = 0; d != 3; ++d) {
      basis(0, d) = r0[d];
      basis(1, d) = r1[d];
    }
    auto btb = basis * basis.transpose();
    auto b = basis * (v0 - v1);
    return inverse(btb) * b;  // beta
  }
  template <typename VecTA, typename VecTB,
            enable_if_all<VecTA::dim == 1, VecTA::extent == 3,
                          is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
  constexpr auto point_triangle_tangent_basis(const VecInterface<VecTA>& v0,
                                              const VecInterface<VecTB>& v1,
                                              const VecInterface<VecTB>& v2,
                                              const VecInterface<VecTB>& v3) {
    using T = typename VecTA::value_type;
    using Ti = typename VecTA::index_type;
    using RetT = typename VecTA::template variant_vec<T, integer_seq<Ti, 3, 2>>;
    RetT basis{};
    auto v12 = v2 - v1;
    auto c0 = v12.normalized();
    auto c1 = v12.cross(v3 - v1).cross(v12).normalized();
    for (int d = 0; d != 3; ++d) {
      basis(d, 0) = c0[d];
      basis(d, 1) = c1[d];
    }
    return basis;
  }

  /// EE
  template <typename VecTA, typename VecTB,
            enable_if_all<VecTA::dim == 1, VecTA::extent == 3,
                          is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
  constexpr auto edge_edge_closest_point(const VecInterface<VecTA>& v0,
                                         const VecInterface<VecTB>& v1,
                                         const VecInterface<VecTB>& v2,
                                         const VecInterface<VecTB>& v3) {
    using T = typename VecTA::value_type;
    using Ti = typename VecTA::index_type;
    using MT = typename VecTA::template variant_vec<T, integer_seq<Ti, 2, 2>>;
    using VT = typename VecTA::template variant_vec<T, integer_seq<Ti, 2>>;
    auto e20 = v0 - v2;
    auto e01 = v1 - v0;
    auto e23 = v3 - v2;
    MT mt{};
    mt(0, 0) = e01.l2NormSqr();
    mt(0, 1) = -e23.dot(e01);
    mt(1, 0) = mt(0, 1);
    mt(1, 1) = e23.l2NormSqr();
    VT vt{};
    vt(0) = -e20.dot(e01);
    vt(1) = e20.dot(e23);
    return inverse(mt) * vt;  // gamma
  }
  template <typename VecTA, typename VecTB,
            enable_if_all<VecTA::dim == 1, VecTA::extent == 3,
                          is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
  constexpr auto edge_edge_tangent_basis(const VecInterface<VecTA>& v0,
                                         const VecInterface<VecTA>& v1,
                                         const VecInterface<VecTB>& v2,
                                         const VecInterface<VecTB>& v3) {
    using T = typename VecTA::value_type;
    using Ti = typename VecTA::index_type;
    using RetT = typename VecTA::template variant_vec<T, integer_seq<Ti, 3, 2>>;
    RetT basis{};
    auto v01 = v1 - v0;
    auto c0 = v01.normalized();
    auto c1 = v01.cross(v3 - v2).cross(v01).normalized();
    for (int d = 0; d != 3; ++d) {
      basis(d, 0) = c0[d];
      basis(d, 1) = c1[d];
    }
    return basis;
  }

}  // namespace zs