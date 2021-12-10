#pragma once
#include "LevelSetInterface.h"
#include "zensim/math/Vec.h"
#include "zensim/types/Polymorphism.h"

namespace zs {

  enum class analytic_geometry_e { Plane, Cuboid, Sphere, Cylinder, Torus };

  template <analytic_geometry_e geomT, typename DataType, int d> struct AnalyticLevelSet;

  template <typename T, int d> struct AnalyticLevelSet<analytic_geometry_e::Plane, T, d>
      : LevelSetInterface<AnalyticLevelSet<analytic_geometry_e::Plane, T, d>> {
    using value_type = T;
    static constexpr int dim = d;
    using TV = vec<value_type, dim>;

    constexpr AnalyticLevelSet() noexcept = default;
    template <typename VecTA, typename VecTB,
              enable_if_all<VecTA::dim == 1, VecTA::extent == dim, VecTB::dim == 1,
                            VecTB::extent == dim> = 0>
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
      return std::make_tuple(_origin, _origin);
    }

    TV _origin{}, _normal{};
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
    constexpr AnalyticLevelSet(const VecInterface<VecT> &center, T len) : _min{}, _max{} {
      _min = center - (len / 2);
      _max = center + (len / 2);
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
      return std::make_tuple(_min, _max);
    }

    TV _min{}, _max{};
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
      return std::make_tuple(_center - _radius, _center + _radius);
    }

    TV _center{};
    T _radius{};
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
          T disL = std::min(_bottom[_d] + _length - x[_d], x[_d] - _bottom[_d]);
          return -std::min(disL, _radius - disR);
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
      auto diffR = TV::uniform(_radius);
      diffR[_d] = (T)0;
      auto diffL = TV::zeros();
      diffL[_d] = _length;
      return std::make_tuple(_bottom - diffR, _bottom + diffR + diffL);
    }

    TV _bottom{};
    T _radius{}, _length{};
    int _d{};
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
