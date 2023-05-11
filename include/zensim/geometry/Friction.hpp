#pragma once

#include "zensim/math/MathUtils.h"
#include "zensim/math/Vec.h"
#include "zensim/math/VecInterface.hpp"

namespace zs {

  /// ref: ipc-sim/Codim-IPC, FEM/FRICTION_UTILS.h
  /// C1
  template <class T> constexpr T f0_SF(T x2, T epsvh) {
    if (x2 >= epsvh * epsvh) {
      return zs::sqrt(x2);
    } else {
      return x2 * (-zs::sqrt(x2) / 3 + epsvh) / (epsvh * epsvh) + epsvh / 3;
    }
  }

  template <class T> constexpr T f1_SF_div_rel_dx_norm(T x2, T epsvh) {
    if (x2 >= epsvh * epsvh) {
      return 1 / zs::sqrt(x2);
    } else {
      return (-zs::sqrt(x2) + 2 * epsvh) / (epsvh * epsvh);
    }
  }

  template <class T> constexpr T f2_SF_term(T x2, T epsvh) {
    return -1 / (epsvh * epsvh);
    // same for x2 >= epsvh * epsvh for C1 clamped friction
  }

  /// PP
  template <typename VecTA, typename VecTB,
            enable_if_all<VecTA::dim == 1, VecTA::extent == 3,
                          is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
  constexpr auto point_point_tangent_basis(const VecInterface<VecTA>& p0,
                                           const VecInterface<VecTB>& p1) {
    using T = typename VecTA::value_type;
    using Ti = typename VecTA::index_type;
    using TV = typename VecTA::template variant_vec<T, integer_sequence<Ti, 3>>;
    using RetT = typename VecTA::template variant_vec<T, integer_sequence<Ti, 3, 2>>;
    RetT basis{};
    auto v01 = (p1 - p0);
    TV xCross{}, yCross{};
    xCross(0) = 0;
    xCross(1) = -v01[2];
    xCross(2) = v01[1];
    yCross(0) = v01[2];
    yCross(1) = 0;
    yCross(2) = -v01[0];
    TV c0{}, c1{};
    if (xCross.l2NormSqr() > yCross.l2NormSqr()) {
      c0 = xCross.normalized();
      c1 = v01.cross(xCross).normalized();
    } else {
      c0 = yCross.normalized();
      c1 = v01.cross(yCross).normalized();
    }
    for (int d = 0; d != 3; ++d) {
      basis(d, 0) = c0[d];
      basis(d, 1) = c1[d];
    }
    return basis;
  }
  template <typename VecTV, enable_if_all<VecTV::dim == 1, VecTV::extent == 3> = 0>
  constexpr auto point_point_rel_dx(const VecInterface<VecTV>& dx0,
                                    const VecInterface<VecTV>& dx1) {
    return dx0 - dx1;  // relDX
  }
  template <typename VecTV, typename VecTM,
            enable_if_all<VecTV::dim == 1, VecTV::extent == 2, VecTM::dim == 2,
                          VecTM::template range_t<0>::value == 3,
                          VecTM::template range_t<1>::value == 2> = 0>
  constexpr auto point_point_rel_dx_tan_to_mesh(const VecInterface<VecTV>& relDXTan,
                                                const VecInterface<VecTM>& basis) {
    using T = typename VecTM::value_type;
    using Ti = typename VecTM::index_type;
    using RetT = typename VecTM::template variant_vec<T, integer_sequence<Ti, 2, 3>>;
    RetT TTTDX{};
    auto val = basis * relDXTan;
    for (Ti d = 0; d != 3; ++d) {
      TTTDX(0, d) = val(d);
      TTTDX(1, d) = -val(d);
    }
    return TTTDX;
  }
  template <typename VecTM, enable_if_all<VecTM::dim == 2, VecTM::template range_t<0>::value == 3,
                                          VecTM::template range_t<1>::value == 2> = 0>
  constexpr auto point_point_TT(const VecInterface<VecTM>& basis) {
    using T = typename VecTM::value_type;
    using Ti = typename VecTM::index_type;
    using HessT = typename VecTM::template variant_vec<T, integer_sequence<Ti, 2, 6>>;
    HessT hess{};
    for (Ti r = 0; r != 2; ++r)
      for (Ti c = 0; c != 3; ++c) {
        hess(r, c) = basis(c, r);
        hess(r, 3 + c) = -basis(c, r);
      }
    return hess;
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
    using RetT = typename VecTA::template variant_vec<T, integer_sequence<Ti, 3, 2>>;
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

  template <typename VecTV, typename T, enable_if_all<VecTV::dim == 1, VecTV::extent == 3> = 0>
  constexpr auto point_edge_rel_dx(const VecInterface<VecTV>& dx0, const VecInterface<VecTV>& dx1,
                                   const VecInterface<VecTV>& dx2, T yita) {
    return dx0 - (dx1 + yita * (dx2 - dx1));
  }
  template <typename VecTV, typename VecTM, typename TT,
            enable_if_all<VecTV::dim == 1, VecTV::extent == 2, VecTM::dim == 2,
                          VecTM::template range_t<0>::value == 3,
                          VecTM::template range_t<1>::value == 2> = 0>
  constexpr auto point_edge_rel_dx_tan_to_mesh(const VecInterface<VecTV>& relDXTan,
                                               const VecInterface<VecTM>& basis, TT yita) {
    using T = typename VecTM::value_type;
    using Ti = typename VecTM::index_type;
    using RetT = typename VecTM::template variant_vec<T, integer_sequence<Ti, 3, 3>>;
    RetT TTTDX{};
    auto val = basis * relDXTan;
    for (Ti d = 0; d != 3; ++d) {
      TTTDX(0, d) = val(d);
      TTTDX(1, d) = (yita - 1) * val(d);
      TTTDX(2, d) = -yita * val(d);
    }
    return TTTDX;
  }
  template <typename VecTM, typename TT,
            enable_if_all<VecTM::dim == 2, VecTM::template range_t<0>::value == 3,
                          VecTM::template range_t<1>::value == 2> = 0>
  constexpr auto point_edge_TT(const VecInterface<VecTM>& basis, TT yita) {
    using T = typename VecTM::value_type;
    using Ti = typename VecTM::index_type;
    using HessT = typename VecTM::template variant_vec<T, integer_sequence<Ti, 2, 9>>;
    HessT hess{};
    for (Ti r = 0; r != 2; ++r)
      for (Ti c = 0; c != 3; ++c) {
        hess(r, c) = basis(c, r);
        hess(r, 3 + c) = (yita - 1) * basis(c, r);
        hess(r, 6 + c) = -yita * basis(c, r);
      }
    return hess;
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
    using MT = typename VecTA::template variant_vec<T, integer_sequence<Ti, 2, 3>>;
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
    using RetT = typename VecTA::template variant_vec<T, integer_sequence<Ti, 3, 2>>;
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
  template <typename VecTV, typename T, enable_if_all<VecTV::dim == 1, VecTV::extent == 3> = 0>
  constexpr auto point_triangle_rel_dx(const VecInterface<VecTV>& dx0,
                                       const VecInterface<VecTV>& dx1,
                                       const VecInterface<VecTV>& dx2,
                                       const VecInterface<VecTV>& dx3, T beta1, T beta2) {
    return dx0 - (dx1 + beta1 * (dx2 - dx1) + beta2 * (dx3 - dx1));
  }
  template <typename VecTV, typename VecTM, typename TT,
            enable_if_all<VecTV::dim == 1, VecTV::extent == 2, VecTM::dim == 2,
                          VecTM::template range_t<0>::value == 3,
                          VecTM::template range_t<1>::value == 2> = 0>
  constexpr auto point_triangle_rel_dx_tan_to_mesh(const VecInterface<VecTV>& relDXTan,
                                                   const VecInterface<VecTM>& basis, TT beta1,
                                                   TT beta2) {
    using T = typename VecTM::value_type;
    using Ti = typename VecTM::index_type;
    using RetT = typename VecTM::template variant_vec<T, integer_sequence<Ti, 4, 3>>;
    RetT TTTDX{};
    auto val = basis * relDXTan;
    for (Ti d = 0; d != 3; ++d) {
      TTTDX(0, d) = val(d);
      TTTDX(1, d) = (-1 + beta1 + beta2) * val(d);
      TTTDX(2, d) = -beta1 * val(d);
      TTTDX(3, d) = -beta2 * val(d);
    }
    return TTTDX;
  }
  template <typename VecTM, typename TT,
            enable_if_all<VecTM::dim == 2, VecTM::template range_t<0>::value == 3,
                          VecTM::template range_t<1>::value == 2> = 0>
  constexpr auto point_triangle_TT(const VecInterface<VecTM>& basis, TT beta1, TT beta2) {
    using T = typename VecTM::value_type;
    using Ti = typename VecTM::index_type;
    using HessT = typename VecTM::template variant_vec<T, integer_sequence<Ti, 2, 12>>;
    HessT hess{};
    for (Ti r = 0; r != 2; ++r)
      for (Ti c = 0; c != 3; ++c) {
        hess(r, c) = basis(c, r);
        hess(r, 3 + c) = (-1 + beta1 + beta2) * basis(c, r);
        hess(r, 6 + c) = -beta1 * basis(c, r);
        hess(r, 9 + c) = -beta2 * basis(c, r);
      }
    return hess;
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
    using MT = typename VecTA::template variant_vec<T, integer_sequence<Ti, 2, 2>>;
    using VT = typename VecTA::template variant_vec<T, integer_sequence<Ti, 2>>;
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
    using RetT = typename VecTA::template variant_vec<T, integer_sequence<Ti, 3, 2>>;
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
  template <typename VecTV, typename T, enable_if_all<VecTV::dim == 1, VecTV::extent == 3> = 0>
  constexpr auto edge_edge_rel_dx(const VecInterface<VecTV>& dx0, const VecInterface<VecTV>& dx1,
                                  const VecInterface<VecTV>& dx2, const VecInterface<VecTV>& dx3,
                                  T gamma1, T gamma2) {
    return dx0 + gamma1 * (dx1 - dx0) - (dx2 + gamma2 * (dx3 - dx2));
  }
  template <typename VecTV, typename VecTM, typename TT,
            enable_if_all<VecTV::dim == 1, VecTV::extent == 2, VecTM::dim == 2,
                          VecTM::template range_t<0>::value == 3,
                          VecTM::template range_t<1>::value == 2> = 0>
  constexpr auto edge_edge_rel_dx_tan_to_mesh(const VecInterface<VecTV>& relDXTan,
                                              const VecInterface<VecTM>& basis, TT gamma1,
                                              TT gamma2) {
    using T = typename VecTM::value_type;
    using Ti = typename VecTM::index_type;
    using RetT = typename VecTM::template variant_vec<T, integer_sequence<Ti, 4, 3>>;
    RetT TTTDX{};
    auto val = basis * relDXTan;
    for (Ti d = 0; d != 3; ++d) {
      TTTDX(0, d) = (1 - gamma1) * val(d);
      TTTDX(1, d) = gamma1 * val(d);
      TTTDX(2, d) = (gamma2 - 1) * val(d);
      TTTDX(3, d) = -gamma2 * val(d);
    }
    return TTTDX;
  }
  template <typename VecTM, typename TT,
            enable_if_all<VecTM::dim == 2, VecTM::template range_t<0>::value == 3,
                          VecTM::template range_t<1>::value == 2> = 0>
  constexpr auto edge_edge_TT(const VecInterface<VecTM>& basis, TT gamma1, TT gamma2) {
    using T = typename VecTM::value_type;
    using Ti = typename VecTM::index_type;
    using HessT = typename VecTM::template variant_vec<T, integer_sequence<Ti, 2, 12>>;
    HessT hess{};
    for (Ti r = 0; r != 2; ++r)
      for (Ti c = 0; c != 3; ++c) {
        hess(r, c) = (1 - gamma1) * basis(c, r);
        hess(r, 3 + c) = gamma1 * basis(c, r);
        hess(r, 6 + c) = (gamma2 - 1) * basis(c, r);
        hess(r, 9 + c) = -gamma2 * basis(c, r);
      }
    return hess;
  }

}  // namespace zs