#pragma once
#include "zensim/geometry/LevelSetInterface.h"
#include "zensim/math/Vec.h"

namespace zs {

  enum class analytic_geometry_e { Plane, Cuboid, Sphere, Cylinder, Torus };

  template <analytic_geometry_e geomT, typename DataType, int d> struct AnalyticLevelSet;

  template <typename T, int d> struct AnalyticLevelSet<analytic_geometry_e::Plane, T, d>
      : LevelSetInterface<AnalyticLevelSet<analytic_geometry_e::Plane, T, d>> {
    using value_type = T;
    static constexpr int dim = d;
    using TV = vec<value_type, dim>;

    constexpr AnalyticLevelSet() noexcept = default;
    template <
        typename VecTA, typename VecTB,
        enable_if_all<VecTA::dim == 1, VecTA::extent == dim, VecTB::dim == 1, VecTB::extent == dim>
        = 0>
    constexpr AnalyticLevelSet(const VecInterface<VecTA> &origin,
                               const VecInterface<VecTB> &normal) noexcept
        : _origin{}, _normal{} {
      _origin = origin;
      _normal = normal;
    }

    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr T do_getSignedDistance(const VecInterface<VecT> &x) const noexcept {
      return _normal.dot(x - _origin);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr TV do_getNormal(const VecInterface<VecT> &x) const noexcept {
      return _normal;
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr TV do_getMaterialVelocity(const VecInterface<VecT> &x) const noexcept {
      return TV::zeros();
    }
    constexpr decltype(auto) do_getBoundingBox() const noexcept {
      return zs::make_tuple(_origin, _origin);
    }

    TV _origin{}, _normal{};

#if ZS_ENABLE_SERIALIZATION
    template <typename S> void serialize(S &s) {
      serialize(s, _origin);
      serialize(s, _normal);
    }
#endif
  };

  template <typename T, int d> struct AnalyticLevelSet<analytic_geometry_e::Cuboid, T, d>
      : LevelSetInterface<AnalyticLevelSet<analytic_geometry_e::Cuboid, T, d>> {
    using value_type = T;
    static constexpr int dim = d;
    using TV = vec<T, dim>;

    constexpr AnalyticLevelSet() noexcept = default;
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr AnalyticLevelSet(const VecInterface<VecT> &min,
                               const VecInterface<VecT> &max) noexcept
        : _min{}, _max{} {
      _min = min;
      _max = max;
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr AnalyticLevelSet(const tuple<VecT, VecT> &bv) noexcept : _min{}, _max{} {
      _min = get<0>(bv);
      _max = get<1>(bv);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr AnalyticLevelSet(const VecInterface<VecT> &center, T len = 0) : _min{}, _max{} {
      _min = center - (len / 2);
      _max = center + (len / 2);
    }
    template <typename... Tis> constexpr auto getVert(Tis... is_) const noexcept {
      static_assert(sizeof...(is_) == dim, "dimension mismtach!");
      int is[] = {is_...};
      TV ret{};
      for (int i = 0; i != sizeof...(is_); ++i) {
        ret[i] = is[i] == 0 ? _min[i] : _max[i];
      }
      return ret;
    }

    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr T do_getSignedDistance(const VecInterface<VecT> &x) const noexcept {
      TV center = (_min + _max) / 2;
      TV point = (x - center).abs() - (_max - _min) / 2;
      T max = point.max();
      for (int i = 0; i != dim; ++i)
        if (point(i) < 0) point(i) = 0;  ///< inside the box
      return (max < 0 ? max : 0) + point.length();
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr TV do_getNormal(const VecInterface<VecT> &x) const noexcept {
      TV diff{}, v1{}, v2{};
      T eps = (T)1e-6;
      /// compute a local partial derivative
      for (int i = 0; i != dim; i++) {
        v1 = x;
        v2 = x;
        v1(i) = x(i) + eps;
        v2(i) = x(i) - eps;
        diff(i) = (getSignedDistance(v1) - getSignedDistance(v2)) / (eps + eps);
      }
      return diff.normalized();
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr TV do_getMaterialVelocity(const VecInterface<VecT> &x) const noexcept {
      return TV::zeros();
    }
    constexpr decltype(auto) do_getBoundingBox() const noexcept {
      return zs::make_tuple(_min, _max);
    }

    TV _min{}, _max{};

#if ZS_ENABLE_SERIALIZATION
    template <typename S> void serialize(S &s) {
      serialize(s, _min);
      serialize(s, _max);
    }
#endif
  };

  template <typename T, int d> struct AnalyticLevelSet<analytic_geometry_e::Sphere, T, d>
      : LevelSetInterface<AnalyticLevelSet<analytic_geometry_e::Sphere, T, d>> {
    using value_type = T;
    static constexpr int dim = d;
    using TV = vec<T, dim>;

    constexpr AnalyticLevelSet() noexcept = default;
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr AnalyticLevelSet(const VecInterface<VecT> &center, T radius) noexcept
        : _center{}, _radius{radius} {
      _center = center;
    }

    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr T do_getSignedDistance(const VecInterface<VecT> &x) const noexcept {
      return (x - _center).length() - _radius;
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr TV do_getNormal(const VecInterface<VecT> &x) const noexcept {
      TV outward_normal = x - _center;
      if (outward_normal.l2NormSqr() < (T)1e-7) return TV::zeros();
      return outward_normal.normalized();
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr TV do_getMaterialVelocity(const VecInterface<VecT> &x) const noexcept {
      return TV::zeros();
    }
    constexpr decltype(auto) do_getBoundingBox() const noexcept {
      return zs::make_tuple(_center - _radius, _center + _radius);
    }

    TV _center{};
    T _radius{};

#if ZS_ENABLE_SERIALIZATION
    template <typename S> void serialize(S &s) {
      serialize(s, _center);
      s.template value<sizeof(T)>(_radius);
    }
#endif
  };

  template <typename T, int d> struct AnalyticLevelSet<analytic_geometry_e::Cylinder, T, d>
      : LevelSetInterface<AnalyticLevelSet<analytic_geometry_e::Cylinder, T, d>> {
    static_assert(d == 3, "dimension of cylinder must be 3");
    using value_type = T;
    static constexpr int dim = d;
    using TV = vec<T, dim>;

    constexpr AnalyticLevelSet() noexcept = default;
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr AnalyticLevelSet(const VecInterface<VecT> &bottom, T radius, T length,
                               int ori) noexcept
        : _bottom{}, _radius{radius}, _length{length}, _d{ori} {
      _bottom = bottom;
    }

    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr T do_getSignedDistance(const VecInterface<VecT> &x) const noexcept {
      vec<T, dim - 1> diffR{};
      for (int k = 0, i = 0; k != dim; ++k)
        if (k != _d) diffR[i++] = x[k] - _bottom[k];
      auto disR = zs::sqrt(diffR.l2NormSqr());
      bool outsideCircle = disR > _radius;

      if (x[_d] < _bottom[_d]) {
        T disL = _bottom[_d] - x[_d];
        if (outsideCircle)
          return zs::sqrt((disR - _radius) * (disR - _radius) + disL * disL);
        else
          return disL;
      } else if (x[_d] > _bottom[_d] + _length) {
        T disL = x[_d] - (_bottom[_d] + _length);
        if (outsideCircle)
          return zs::sqrt((disR - _radius) * (disR - _radius) + disL * disL);
        else
          return disL;
      } else {
        if (outsideCircle)
          return disR - _radius;
        else {
          T disL = math::min(_bottom[_d] + _length - x[_d], x[_d] - _bottom[_d]);
          return -math::min(disL, _radius - disR);
        }
      }
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr TV do_getNormal(const VecInterface<VecT> &x) const noexcept {
      TV diff{}, v1{}, v2{};
      T eps = (T)1e-6;
      /// compute a local partial derivative
      for (int i = 0; i != dim; i++) {
        v1 = x;
        v2 = x;
        v1(i) = x(i) + eps;
        v2(i) = x(i) - eps;
        diff(i) = (getSignedDistance(v1) - getSignedDistance(v2)) / (eps + eps);
      }
      return diff.normalized();
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr TV do_getMaterialVelocity(const VecInterface<VecT> &x) const noexcept {
      return TV::zeros();
    }
    constexpr decltype(auto) do_getBoundingBox() const noexcept {
      auto diffR = TV::constant(_radius);
      diffR[_d] = (T)0;
      auto diffL = TV::zeros();
      diffL[_d] = _length;
      return zs::make_tuple(_bottom - diffR, _bottom + diffR + diffL);
    }

    TV _bottom{};
    T _radius{}, _length{};
    int _d{};

#if ZS_ENABLE_SERIALIZATION
    template <typename S> void serialize(S &s) {
      serialize(s, _bottom);
      s.template value<sizeof(T)>(_radius);
      s.template value<sizeof(T)>(_length);
      s.template value<sizeof(int)>(_d);
    }
#endif
  };

  /// Bounding Volume
  /// AABBBox
  template <int dim, typename T = float> using AABBBox
      = AnalyticLevelSet<analytic_geometry_e::Cuboid, T, dim>;

  template <int dim, typename T>
  constexpr bool overlaps(const AABBBox<dim, T> &a, const AABBBox<dim, T> &b) noexcept {
    for (int d = 0; d < dim; ++d)
      if (b._min[d] > a._max[d] || b._max[d] < a._min[d]) return false;
    return true;
  }
  template <int dim, typename T>
  constexpr bool overlaps(const vec<T, dim> &p, const AABBBox<dim, T> &b) noexcept {
    for (int d = 0; d < dim; ++d)
      if (b._min[d] > p[d] || b._max[d] < p[d]) return false;
    return true;
  }
  template <int dim, typename T>
  constexpr bool overlaps(const AABBBox<dim, T> &b, const vec<T, dim> &p) noexcept {
    for (int d = 0; d < dim; ++d)
      if (b._min[d] > p[d] || b._max[d] < p[d]) return false;
    return true;
  }

  template <int dim, typename T, typename VecT,
            enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
  constexpr void merge(AABBBox<dim, T> &box, const VecInterface<VecT> &p) noexcept {
    using TT = math::op_result_t<T, typename VecT::value_type>;
    for (int d = 0; d != dim; ++d) {
      box._min[d] = zs::min(static_cast<TT>(box._min[d]), static_cast<TT>(p[d]));
      box._max[d] = zs::max(static_cast<TT>(box._max[d]), static_cast<TT>(p[d]));
      // if (p[d] < box._min[d]) box._min[d] = p[d];
      // if (p[d] > box._max[d]) box._max[d] = p[d];
    }
  }

  template <int dim, typename T_, typename VecT,
            enable_if_all<VecT::dim == 1, VecT::extent == dim, (dim > 0)> = 0>
  constexpr auto distance(const AABBBox<dim, T_> &b, const VecInterface<VecT> &p) noexcept {
    using T = math::op_result_t<T_, typename VecT::value_type>;
    const auto &[mi, ma] = b;
    auto center = (mi + ma) / 2;
    auto point = (p - center).abs() - (ma - mi) / 2;
    T max = detail::deduce_numeric_lowest<T>();
    for (int d = 0; d != dim; ++d) {
      if (point[d] > max) max = point[d];
      if (point[d] < 0) point[d] = 0;
    }
    return (max < 0 ? max : 0) + point.length();
  }

  template <typename VecT, int dim, typename T>
  constexpr auto distance(const VecInterface<VecT> &p, const AABBBox<dim, T> &b) noexcept {
    return distance(b, p);
  }

  template <typename VecT>
  constexpr bool pt_ccd_broadphase(const VecInterface<VecT> &p, const VecInterface<VecT> &t0,
                                   const VecInterface<VecT> &t1, const VecInterface<VecT> &t2,
                                   const VecInterface<VecT> &dp, const VecInterface<VecT> &dt0,
                                   const VecInterface<VecT> &dt1, const VecInterface<VecT> &dt2,
                                   const typename VecT::value_type dist,
                                   const typename VecT::value_type toc_upperbound) {
    constexpr int dim = VecT::template range_t<0>::value;
    using T = typename VecT::value_type;
    using bv_t = AABBBox<dim, T>;
    bv_t pbv{get_bounding_box(p, p + toc_upperbound * dp)},
        tbv{get_bounding_box(t0, t0 + toc_upperbound * dt0)};
    merge(tbv, t1);
    merge(tbv, t1 + toc_upperbound * dt1);
    merge(tbv, t2);
    merge(tbv, t2 + toc_upperbound * dt2);
    pbv._min -= dist;
    pbv._max += dist;
    return overlaps(pbv, tbv);
  }

  template <typename VecT>
  constexpr bool ee_ccd_broadphase(const VecInterface<VecT> &ea0, const VecInterface<VecT> &ea1,
                                   const VecInterface<VecT> &eb0, const VecInterface<VecT> &eb1,
                                   const VecInterface<VecT> &dea0, const VecInterface<VecT> &dea1,
                                   const VecInterface<VecT> &deb0, const VecInterface<VecT> &deb1,
                                   const typename VecT::value_type dist,
                                   const typename VecT::value_type toc_upperbound) {
    constexpr int dim = VecT::template range_t<0>::value;
    using T = typename VecT::value_type;
    using bv_t = AABBBox<dim, T>;
    bv_t abv{get_bounding_box(ea0, ea0 + toc_upperbound * dea0)},
        bbv{get_bounding_box(eb0, eb0 + toc_upperbound * deb0)};
    merge(abv, ea1);
    merge(abv, ea1 + toc_upperbound * dea1);
    merge(bbv, eb1);
    merge(bbv, eb1 + toc_upperbound * deb1);
    abv._min -= dist;
    abv._max += dist;
    return overlaps(abv, bbv);
  }

  template <typename VecT> constexpr typename VecT::value_type ray_tri_intersect(
      VecInterface<VecT> const &ro, VecInterface<VecT> const &rd, VecInterface<VecT> const &v0,
      VecInterface<VecT> const &v1, VecInterface<VecT> const &v2) {
    using T = typename VecT::value_type;
    constexpr T eps = detail::deduce_numeric_epsilon<T>() * 10;
    auto u = v1 - v0;
    auto v = v2 - v0;
    auto n = u.cross(v);
    T b = n.dot(rd);
    if (zs::abs(b) > eps) {
      T a = n.dot(v0 - ro);
      T r = a / b;
      {  // could provide predicates here
        auto ip = ro + r * rd;
        T uu = u.dot(u);
        T uv = u.dot(v);
        T vv = v.dot(v);
        auto w = ip - v0;
        T wu = w.dot(u);
        T wv = w.dot(v);
        T d = uv * uv - uu * vv;
        T s = uv * wv - vv * wu;
        T t = uv * wu - uu * wv;
        d = (T)1 / d;
        s *= d;
        t *= d;
        if (-eps <= s && s <= 1 + eps && -eps <= t && s + t <= 1 + eps * 2) return r;
      }
    }
    return detail::deduce_numeric_infinity<T>();
  }

  /// ref: An Efficient and Robust Ray-Box Intersection Algorithm, 2005
  template <typename VecT, int dim, typename T_,
            enable_if_all<VecT::dim == 1, VecT::extent == 3> = 0>
  constexpr bool ray_box_intersect(VecInterface<VecT> const &ro, VecInterface<VecT> const &rd,
                                   AABBBox<dim, T_> const &box) noexcept {
    static_assert(is_floating_point_v<typename VecT::value_type>,
                  "ray direction should be in floating point");
    using T = math::op_result_t<typename VecT::value_type, T_>;
    // if (rd.l2NormSqr() < detail::deduce_numeric_epsilon<T>() * 10) return false;
    T invd[3] = {1 / rd[0], 1 / rd[1], 1 / rd[2]};  // allow div 0, assuming IEEE standard
    int sign[3] = {invd[0] < 0, invd[1] < 0, invd[2] < 0};
    T tmin{}, tmax{}, tymin{}, tymax{}, tzmin{}, tzmax{};

    tmin = ((sign[0] ? box._max : box._min)[0] - ro[0]) * invd[0];
    tmax = ((sign[0] ? box._min : box._max)[0] - ro[0]) * invd[0];
    tymin = ((sign[1] ? box._max : box._min)[1] - ro[1]) * invd[1];
    tymax = ((sign[1] ? box._min : box._max)[1] - ro[1]) * invd[1];
    if (tmin > tymax || tymin > tmax) return false;
    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;
    tzmin = ((sign[2] ? box._max : box._min)[2] - ro[2]) * invd[2];
    tzmax = ((sign[2] ? box._min : box._max)[2] - ro[2]) * invd[2];
    if (tmin > tzmax || tzmin > tmax) return false;
    if (tzmin > tmin) tmin = tzmin;
    if (tzmax < tmax) tmax = tzmax;
    return tmax >= (T)0;
  }

  /// Sphere
  template <int dim, typename T = float> using BoundingSphere
      = AnalyticLevelSet<analytic_geometry_e::Sphere, T, dim>;

  template <int dim, typename T> constexpr bool overlaps(const BoundingSphere<dim, T> &a,
                                                         const BoundingSphere<dim, T> &b) noexcept {
    auto radius = a._radius + b._radius;
    auto disSqr = (a._center - b._center).l2NormSqr();
    return disSqr <= radius * radius;
  }
  template <int dim, typename T>
  constexpr bool overlaps(const vec<T, dim> &p, const BoundingSphere<dim, T> &b) noexcept {
    auto radius = b._radius;
    auto disSqr = (p - b._center).l2NormSqr();
    return disSqr <= radius * radius;
  }
  template <int dim, typename T>
  constexpr bool overlaps(const BoundingSphere<dim, T> &b, const vec<T, dim> &p) noexcept {
    auto radius = b._radius;
    auto disSqr = (p - b._center).l2NormSqr();
    return disSqr <= radius * radius;
  }

}  // namespace zs
