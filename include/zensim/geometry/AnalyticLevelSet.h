#pragma once
#include "LevelSetInterface.h"
#include "zensim/math/Vec.h"
#include "zensim/types/Polymorphism.h"

namespace zs {

  enum class analytic_geometry_e { Plane, Cuboid, Sphere, Cylinder, Torus };

  template <analytic_geometry_e geomT, typename DataType, int d> struct AnalyticLevelSet;

  template <typename DataType, int d>
  struct AnalyticLevelSet<analytic_geometry_e::Plane, DataType, d>
      : LevelSetInterface<AnalyticLevelSet<analytic_geometry_e::Plane, DataType, d>, DataType, d> {
    using T = DataType;
    static constexpr int dim = d;
    using TV = vec<T, dim>;

    AnalyticLevelSet() = default;
    ~AnalyticLevelSet() = default;
    constexpr AnalyticLevelSet(TV origin, TV normal)
        : _origin{origin}, _normal{normal.normalized()} {}

    constexpr T getSignedDistance(const TV &X) const noexcept { return _normal.dot(X - _origin); }
    constexpr TV getNormal(const TV &X) const noexcept { return _normal; }
    constexpr TV getMaterialVelocity(const TV &X) const noexcept { return TV::zeros(); }
    constexpr decltype(auto) do_getBoundingBox() const noexcept {
      return std::make_tuple(_origin, _origin);
    }

    TV _origin{}, _normal{};
  };

  template <typename DataType, int d>
  struct AnalyticLevelSet<analytic_geometry_e::Cuboid, DataType, d>
      : LevelSetInterface<AnalyticLevelSet<analytic_geometry_e::Cuboid, DataType, d>, DataType, d> {
    using T = DataType;
    static constexpr int dim = d;
    using TV = vec<T, dim>;

    AnalyticLevelSet() = default;
    ~AnalyticLevelSet() = default;
    constexpr AnalyticLevelSet(TV min, TV max) : _min{min}, _max{max} {}
    constexpr AnalyticLevelSet(TV center, T len)
        : _min{center - (len / 2)}, _max{center + (len / 2)} {}

    constexpr T getSignedDistance(const TV &X) const noexcept {
      TV center = (_min + _max) / 2;
      TV point = (X - center).abs() - (_max - _min) / 2;
      T max = point.max();
      for (int i = 0; i < dim; ++i)
        if (point(i) < 0) point(i) = 0;  ///< inside the box
      return (max < 0 ? max : 0) + point.length();
    }
    constexpr TV getNormal(const TV &X) const noexcept {
      TV diff{}, v1{}, v2{};
      T eps = (T)1e-6;
      /// compute a local partial derivative
      for (int i = 0; i < dim; i++) {
        v1 = X;
        v2 = X;
        v1(i) = X(i) + eps;
        v2(i) = X(i) - eps;
        diff(i) = (getSignedDistance(v1) - getSignedDistance(v2)) / (eps + eps);
      }
      return diff.normalized();
    }
    constexpr TV getMaterialVelocity(const TV &X) const noexcept { return TV::zeros(); }
    constexpr decltype(auto) do_getBoundingBox() const noexcept {
      return std::make_tuple(_min, _max);
    }

    TV _min{}, _max{};
  };

  template <typename DataType, int d>
  struct AnalyticLevelSet<analytic_geometry_e::Sphere, DataType, d>
      : LevelSetInterface<AnalyticLevelSet<analytic_geometry_e::Sphere, DataType, d>, DataType, d> {
    using T = DataType;
    static constexpr int dim = d;
    using TV = vec<T, dim>;

    AnalyticLevelSet() = default;
    ~AnalyticLevelSet() = default;
    constexpr AnalyticLevelSet(TV center, T radius) : _center{center}, _radius{radius} {}

    constexpr T getSignedDistance(const TV &X) const noexcept {
      return (X - _center).length() - _radius;
    }
    constexpr TV getNormal(const TV &X) const noexcept {
      TV outward_normal = X - _center;
      if (outward_normal.l2NormSqr() < (T)1e-7) return TV::zeros();
      return outward_normal.normalized();
    }
    constexpr TV getMaterialVelocity(const TV &X) const noexcept { return TV::zeros(); }
    constexpr decltype(auto) do_getBoundingBox() const noexcept {
      return std::make_tuple(_center - _radius, _center + _radius);
    }

    TV _center{};
    T _radius{};
  };

  template <typename DataType, int d>
  struct AnalyticLevelSet<analytic_geometry_e::Cylinder, DataType, d>
      : LevelSetInterface<AnalyticLevelSet<analytic_geometry_e::Cylinder, DataType, d>, DataType,
                          d> {
    static_assert(d == 3, "dimension of cylinder must be 3");
    using T = DataType;
    static constexpr int dim = d;
    using TV = vec<T, dim>;

    AnalyticLevelSet() = default;
    ~AnalyticLevelSet() = default;
    constexpr AnalyticLevelSet(TV bottom, T radius, T length, int ori)
        : _bottom{bottom}, _radius{radius}, _length{length}, _d{ori} {}

    constexpr T getSignedDistance(const TV &X) const noexcept {
      vec<T, dim - 1> diffR{};
      for (int k = 0, i = 0; k != dim; ++k)
        if (k != _d) diffR[i++] = X[k] - _bottom[k];
      auto disR = zs::sqrt(diffR.l2NormSqr());
      bool outsideCircle = disR > _radius;

      if (X[_d] < _bottom[_d]) {
        T disL = _bottom[_d] - X[_d];
        if (outsideCircle)
          return zs::sqrt((disR - _radius) * (disR - _radius) + disL * disL);
        else
          return disL;
      } else if (X[_d] > _bottom[_d] + _length) {
        T disL = X[_d] - (_bottom[_d] + _length);
        if (outsideCircle)
          return zs::sqrt((disR - _radius) * (disR - _radius) + disL * disL);
        else
          return disL;
      } else {
        if (outsideCircle)
          return disR - _radius;
        else {
          T disL = std::min(_bottom[_d] + _length - X[_d], X[_d] - _bottom[_d]);
          return -std::min(disL, _radius - disR);
        }
      }
    }
    constexpr TV getNormal(const TV &X) const noexcept {
      TV diff{}, v1{}, v2{};
      T eps = (T)1e-6;
      /// compute a local partial derivative
      for (int i = 0; i < dim; i++) {
        v1 = X;
        v2 = X;
        v1(i) = X(i) + eps;
        v2(i) = X(i) - eps;
        diff(i) = (getSignedDistance(v1) - getSignedDistance(v2)) / (eps + eps);
      }
      return diff.normalized();
    }
    constexpr TV getMaterialVelocity(const TV &X) const noexcept { return TV::zeros(); }
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
