#pragma once
#include "zensim/math/VecInterface.hpp"

namespace zs {

  // ref: ipc-tools

  //! point-point
  template <
      typename VecTA, typename VecTB,
      enable_if_all<VecTA::dim == 1, is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
  constexpr auto dist_pp_sqr(const VecInterface<VecTA> &a, const VecInterface<VecTB> &b) noexcept {
    return (b - a).l2NormSqr();
  }
  template <
      typename VecTA, typename VecTB,
      enable_if_all<VecTA::dim == 1, is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
  constexpr auto dist_pp(const VecInterface<VecTA> &a, const VecInterface<VecTB> &b) noexcept {
    return zs::sqrt(dist_pp_sqr(a, b));
  }

  //! point-edge
  template <
      typename VecTP, typename VecTE,
      enable_if_all<VecTP::dim == 1, is_same_v<typename VecTP::dims, typename VecTE::dims>> = 0>
  constexpr int dist_pe_category(const VecInterface<VecTP> &p, const VecInterface<VecTE> &e0,
                                 const VecInterface<VecTE> &e1) noexcept {
    const auto e = e1 - e0;
    auto indicator = e.dot(p - e0) / e.l2NormSqr();
    return indicator < 0 ? 0 : (indicator > 1 ? 1 : 2);
  }

  template <
      typename VecTP, typename VecTE,
      enable_if_all<VecTP::dim == 1, is_same_v<typename VecTP::dims, typename VecTE::dims>> = 0>
  constexpr auto dist_pe_sqr(const VecInterface<VecTP> &p, const VecInterface<VecTE> &e0,
                             const VecInterface<VecTE> &e1) noexcept {
    using T = math::op_result_t<typename VecTP::value_type, typename VecTE::value_type>;
    constexpr int dim = VecTP::extent;
    T ret = limits<T>::max();
    switch (dist_pe_category(p, e0, e1)) {
      case 0:
        ret = dist_pp_sqr(p, e0);
        break;
      case 1:
        ret = dist_pp_sqr(p, e1);
        break;
      case 2:
        if constexpr (dim == 2) {
          const auto e = e1 - e0;
          auto numerator = e[1] * p[0] - e[0] * p[1] + e1[0] * e0[1] - e1[1] * e0[0];
          ret = numerator * numerator / e.l2NormSqr();
        } else if constexpr (dim == 3) {
          ret = cross(e0 - p, e1 - p).l2NormSqr() / (e1 - e0).l2NormSqr();
        }
        break;
      default:
        break;
    }
    return ret;
  }
  template <
      typename VecTP, typename VecTE,
      enable_if_all<VecTP::dim == 1, is_same_v<typename VecTP::dims, typename VecTE::dims>> = 0>
  constexpr auto dist_pe(const VecInterface<VecTP> &p, const VecInterface<VecTE> &e0,
                         const VecInterface<VecTE> &e1) noexcept {
    return zs::sqrt(dist_pe_sqr(p, e0, e1));
  }

  //! point-triangle
  // David Eberly, Geometric Tools, Redmond WA 98052
  // Copyright (c) 1998-2022
  // Distributed under the Boost Software License, Version 1.0.
  // https://www.boost.org/LICENSE_1_0.txt
  // https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
  // Version: 6.0.2022.01.06

  // ref: https://www.geometrictools.com/GTE/Mathematics/DistPointTriangle.h
  template <
      typename VecTP, typename VecTT,
      enable_if_all<VecTP::dim == 1, is_same_v<typename VecTP::dims, typename VecTT::dims>> = 0>
  constexpr auto dist_pt_sqr(const VecInterface<VecTP> &p, const VecInterface<VecTT> &t0,
                             const VecInterface<VecTT> &t1,
                             const VecInterface<VecTT> &t2) noexcept {
    using T = math::op_result_t<typename VecTP::value_type, typename VecTT::value_type>;
    static_assert(std::is_floating_point_v<T>,
                  "value_types of VecTs cannot be both integral type.");
    using TV = typename VecTP::template variant_vec<T, typename VecTP::extents>;
    TV diff = t0 - p;
    TV e0 = t1 - t0;
    TV e1 = t2 - t0;
    T a00 = e0.dot(e0);
    T a01 = e0.dot(e1);
    T a11 = e1.dot(e1);
    T b0 = diff.dot(e0);
    T b1 = diff.dot(e1);
    T det = zs::max(a00 * a11 - a01 * a01, (T)0);
    T s = a01 * b1 - a11 * b0;
    T t = a01 * b0 - a00 * b1;

    if (s + t <= det) {
      if (s < (T)0) {
        if (t < (T)0) {  // region 4
          if (b0 < (T)0) {
            t = (T)0;
            if (-b0 >= a00)
              s = (T)1;
            else
              s = -b0 / a00;
          } else {
            s = (T)0;
            if (b1 >= (T)0)
              t = (T)0;
            else if (-b1 >= a11)
              t = (T)1;
            else
              t = -b1 / a11;
          }
        } else {  // region 3
          s = (T)0;
          if (b1 >= (T)0)
            t = (T)0;
          else if (-b1 >= a11)
            t = (T)1;
          else
            t = -b1 / a11;
        }
      } else if (t < (T)0) {  // region 5
        t = (T)0;
        if (b0 >= (T)0)
          s = (T)0;
        else if (-b0 >= a00)
          s = (T)1;
        else
          s = -b0 / a00;
      } else {  // region 0
                // minimum at interior point
        s /= det;
        t /= det;
      }
    } else {
      T tmp0{}, tmp1{}, numer{}, denom{};
      if (s < (T)0) {  // region 2
        tmp0 = a01 + b0;
        tmp1 = a11 + b1;
        if (tmp1 > tmp0) {
          numer = tmp1 - tmp0;
          denom = a00 - (a01 + a01) + a11;
          if (numer >= denom) {
            s = (T)1;
            t = (T)0;
          } else {
            s = numer / denom;
            t = (T)1 - s;
          }
        } else {
          s = (T)0;
          if (tmp1 <= (T)0)
            t = (T)1;
          else if (b1 >= (T)0)
            t = (T)0;
          else
            t = -b1 / a11;
        }
      } else if (t < (T)0) {  // region 6
        tmp0 = a01 + b1;
        tmp1 = a00 + b0;
        if (tmp1 > tmp0) {
          numer = tmp1 - tmp0;
          denom = a00 - (a01 + a01) + a11;
          if (numer >= denom) {
            t = (T)1;
            s = (T)0;
          } else {
            t = numer / denom;
            s = (T)1 - t;
          }
        } else {
          t = (T)0;
          if (tmp1 <= (T)0)
            s = (T)1;
          else if (b0 >= (T)0)
            s = (T)0;
          else
            s = -b0 / a00;
        }
      } else {  // region 1
        numer = a11 + b1 - a01 - b0;
        if (numer <= (T)0) {
          s = (T)0;
          t = (T)1;
        } else {
          denom = a00 - (a01 + a01) + a11;
          if (numer >= denom) {
            s = (T)1;
            t = (T)0;
          } else {
            s = numer / denom;
            t = (T)1 - s;
          }
        }
      }
    }
    auto hitpoint = t0 + s * e0 + t * e1;
    return (p - hitpoint).l2NormSqr();
  }
  template <
      typename VecTP, typename VecTT,
      enable_if_all<VecTP::dim == 1, is_same_v<typename VecTP::dims, typename VecTT::dims>> = 0>
  constexpr auto dist_pt(const VecInterface<VecTP> &p, const VecInterface<VecTT> &t0,
                         const VecInterface<VecTT> &t1, const VecInterface<VecTT> &t2) noexcept {
    return zs::sqrt(dist_pt_sqr(p, t0, t1, t2));
  }

  // edge-edge
  // ref: <<practical geometry algorithms>> - Daniel Sunday
  // ref: http://geomalgorithms.com/a07-_distance.html
  // ref: dist3D_Segment_to_Segment()
  template <
      typename VecTA, typename VecTB,
      enable_if_all<VecTA::dim == 1, is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
  constexpr auto dist_ee_sqr(const VecInterface<VecTA> &ea0, const VecInterface<VecTA> &ea1,
                             const VecInterface<VecTB> &eb0,
                             const VecInterface<VecTB> &eb1) noexcept {
    using T = math::op_result_t<typename VecTA::value_type, typename VecTB::value_type>;
    auto u = ea1 - ea0;
    auto v = eb1 - eb0;
    auto w = ea0 - eb0;
    float a = u.dot(u);  // >= 0
    float b = u.dot(v);
    float c = v.dot(v);  // >= 0
    float d = u.dot(w);
    float e = v.dot(w);
    float D = a * c - b * b;   // >= 0
    float sc{}, sN{}, sD = D;  // sc = sN/sD
    float tc{}, tN{}, tD = D;

    constexpr auto eps = (T)128 * limits<T>::epsilon();
    if (D < eps) {
      sN = (T)0;
      sD = (T)1;
      tN = e;
      tD = c;
    } else {  // get the closest points on the infinite lines
      sN = b * e - c * d;
      tN = a * e - b * d;
      if (sN < (T)0) {
        sN = (T)0;
        tN = e;
        tD = c;
      } else if (sN > sD) {
        sN = sD;
        tN = e + b;
        tD = c;
      }
    }

    if (tN < (T)0) {
      tN = (T)0;
      if (auto _d = -d; _d < (T)0)
        sN = (T)0;
      else if (_d > a)
        sN = sD;
      else {
        sN = _d;
        sD = a;
      }
    } else if (tN > tD) {
      tN = tD;
      if (auto b_d = -d + b; b_d < (T)0)
        sN = (T)0;
      else if (b_d > a)
        sN = sD;
      else {
        sN = b_d;
        sD = a;
      }
    }

    sc = zs::abs(sN) < eps ? (T)0 : sN / sD;
    tc = zs::abs(tN) < eps ? (T)0 : tN / tD;

    auto dP = w + (sc * u) - (tc * v);
    return dP.l2NormSqr();
  }
  template <
      typename VecTA, typename VecTB,
      enable_if_all<VecTA::dim == 1, is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
  constexpr auto dist_ee(const VecInterface<VecTA> &ea0, const VecInterface<VecTA> &ea1,
                         const VecInterface<VecTB> &eb0, const VecInterface<VecTB> &eb1) noexcept {
    return zs::sqrt(dist_ee_sqr(ea0, ea1, eb0, eb1));
  }

  /// ref: ipc-sim/Codim-IPC, Math/Distance/EDGE_EDGE.h
  namespace detail {
    template <class T, typename VecT> constexpr void g_EE(T v01, T v02, T v03, T v11, T v12, T v13,
                                                          T v21, T v22, T v23, T v31, T v32, T v33,
                                                          VecInterface<VecT> &g) noexcept {
      T t11{};
      T t12{};
      T t13{};
      T t14{};
      T t15{};
      T t16{};
      T t17{};
      T t18{};
      T t19{};
      T t32{};
      T t33{};
      T t34{};
      T t35{};
      T t36{};
      T t37{};
      T t44{};
      T t45{};
      T t46{};
      T t75{};
      T t77{};
      T t76{};
      T t78{};
      T t79{};
      T t80{};
      T t81{};
      T t83{};

      /* G_EE */
      /*     G = G_EE(V01,V02,V03,V11,V12,V13,V21,V22,V23,V31,V32,V33) */
      /*     This function was generated by the Symbolic Math Toolbox version 8.3. */
      /*     14-Jun-2019 13:58:25 */
      t11 = -v11 + v01;
      t12 = -v12 + v02;
      t13 = -v13 + v03;
      t14 = -v21 + v01;
      t15 = -v22 + v02;
      t16 = -v23 + v03;
      t17 = -v31 + v21;
      t18 = -v32 + v22;
      t19 = -v33 + v23;
      t32 = t14 * t18;
      t33 = t15 * t17;
      t34 = t14 * t19;
      t35 = t16 * t17;
      t36 = t15 * t19;
      t37 = t16 * t18;
      t44 = t11 * t18 + -(t12 * t17);
      t45 = t11 * t19 + -(t13 * t17);
      t46 = t12 * t19 + -(t13 * t18);
      t75 = 1.0 / ((t44 * t44 + t45 * t45) + t46 * t46);
      t77 = (t16 * t44 + t14 * t46) + -(t15 * t45);
      t76 = t75 * t75;
      t78 = t77 * t77;
      t79 = (t12 * t44 * 2.0 + t13 * t45 * 2.0) * t76 * t78;
      t80 = (t11 * t45 * 2.0 + t12 * t46 * 2.0) * t76 * t78;
      t81 = (t18 * t44 * 2.0 + t19 * t45 * 2.0) * t76 * t78;
      t18 = (t17 * t45 * 2.0 + t18 * t46 * 2.0) * t76 * t78;
      t83 = (t11 * t44 * 2.0 + -(t13 * t46 * 2.0)) * t76 * t78;
      t19 = (t17 * t44 * 2.0 + -(t19 * t46 * 2.0)) * t76 * t78;
      t76 = t75 * t77;
      g.val(0) = -t81 + t76 * ((-t36 + t37) + t46) * 2.0;
      g.val(1) = t19 - t76 * ((-t34 + t35) + t45) * 2.0;
      g.val(2) = t18 + t76 * ((-t32 + t33) + t44) * 2.0;
      g.val(3) = t81 + t76 * (t36 - t37) * 2.0;
      g.val(4) = -t19 - t76 * (t34 - t35) * 2.0;
      g.val(5) = -t18 + t76 * (t32 - t33) * 2.0;
      t17 = t12 * t16 + -(t13 * t15);
      g.val(6) = t79 - t76 * (t17 + t46) * 2.0;
      t18 = t11 * t16 + -(t13 * t14);
      g.val(7) = -t83 + t76 * (t18 + t45) * 2.0;
      t19 = t11 * t15 + -(t12 * t14);
      g.val(8) = -t80 - t76 * (t19 + t44) * 2.0;
      g.val(9) = -t79 + t76 * t17 * 2.0;
      g.val(10) = t83 - t76 * t18 * 2.0;
      g.val(11) = t80 + t76 * t19 * 2.0;
    }
  }  // namespace detail
  template <
      typename VecTA, typename VecTB,
      enable_if_all<VecTA::dim == 1, is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
  constexpr auto dist2_ee(const VecInterface<VecTA> &ea0, const VecInterface<VecTA> &ea1,
                          const VecInterface<VecTB> &eb0, const VecInterface<VecTB> &eb1) noexcept {
    using T = math::op_result_t<typename VecTA::value_type, typename VecTB::value_type>;
    auto b = (ea1 - ea0).cross(eb1 - eb0);
    T aTb = (eb0 - ea0).dot(b);
    return aTb * aTb / b.l2NormSqr();
  }

  template <
      typename VecTA, typename VecTB,
      enable_if_all<VecTA::dim == 1, is_same_v<typename VecTA::dims, typename VecTB::dims>> = 0>
  constexpr auto dist_grad_ee(const VecInterface<VecTA> &ea0, const VecInterface<VecTA> &ea1,
                              const VecInterface<VecTB> &eb0,
                              const VecInterface<VecTB> &eb1) noexcept {
    using T = math::op_result_t<typename VecTA::value_type, typename VecTB::value_type>;
    using Ti = typename VecTA::index_type;
    constexpr int dim = VecTA::template range_t<0>::value;
    static_assert(dim == 3, "currently only implement 3d version");
    using RetT = typename VecTA::template variant_vec<T, integer_seq<Ti, 4, dim>>;
    RetT ret{};
    detail::g_EE(ea0[0], ea0[1], ea0[2], ea1[0], ea1[1], ea1[2], eb0[0], eb0[1], eb0[2], eb1[0],
                 eb1[1], eb1[2], ret);
    return ret;
  }

  template <typename T> constexpr T barrier_gradient(const T d2, const T dHat2, const T kappa) {
    T grad = 0.0;
    if (d2 < dHat2) {
      T t2 = d2 - dHat2;
      grad = kappa
             * ((t2 / dHat2) * zs::log(d2 / dHat2) * (T)-2.0 / dHat2
                - ((t2 / dHat2) * (t2 / dHat2)) / d2);
    }
    return grad;
  }

}  // namespace zs