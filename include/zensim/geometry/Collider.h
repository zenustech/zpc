#pragma once
#include <zensim/types/Polymorphism.h>

#include <zensim/math/Rotation.hpp>

#include "GenericLevelSet.h"

namespace zs {

  enum class collider_e { Sticky, Slip, Separate };

  template <typename LS> struct Collider {
    using T = typename LS::T;
    static constexpr int dim = LS::dim;
    using TV = vec<T, dim>;

    constexpr void setCollisionType(collider_e ct) noexcept { type = ct; }
    constexpr void setTranslation(TV b_in, TV dbdt_in) noexcept {
      b = b_in;
      dbdt = dbdt_in;
    }
    constexpr void setRotation(Rotation<T, dim> R_in, AngularVelocity<T, dim> omega_in) noexcept {
      R = R_in;
      omega = omega_in;
    }
    constexpr bool queryInside(const TV &x) const noexcept {
      TV x_minus_b = x - b;
      T one_over_s = 1 / s;
      TV X = R.transpose() * x_minus_b * one_over_s;  // material space
      return levelset.getSignedDistance(X) < 0;
    }
    constexpr bool resolveCollision(const TV &x, TV &v, T erosion = 0) const noexcept {
      /** derivation:
          x = \phi(X,t) = R(t)s(t)X+b(t)
          X = \phi^{-1}(x,t) = (1/s) R^{-1} (x-b)
          V(X,t) = \frac{\partial \phi}{\partial t}
                = R'sX + Rs'X + RsX' + b'
          v(x,t) = V(\phi^{-1}(x,t),t)
                = R'R^{-1}(x-b) + (s'/s)(x-b) + RsX' + b'
                = omega \cross (x-b) + (s'/s)(x-b) +b'
      */
      /// collision
      TV x_minus_b = x - b;
      T one_over_s = 1 / s;
      TV X = R.transpose() * x_minus_b * one_over_s;  // material space
      if (levelset.getSignedDistance(X) < -erosion) {
        TV v_object = omega.cross(x_minus_b) + (dsdt * one_over_s) * x_minus_b
                      + R * s * levelset.getMaterialVelocity(X) + dbdt;
        if (type == collider_e::Sticky)
          v = TV::zeros();
        else {
          v -= v_object;
          TV n = R * levelset.getNormal(X);
          T proj = n.dot(v);
          if ((type == collider_e::Separate && proj < 0) || type == collider_e::Slip) v -= proj * n;
          v += v_object;
        }
        return true;
      }
      return false;
    }

    // levelset
    LS levelset;
    collider_e type{collider_e::Sticky};  ///< runtime
    /** scale **/
    T s{1};
    T dsdt{0};
    /** rotation **/
    Rotation<T, dim> R{};
    AngularVelocity<T, dim> omega{};
    /** translation **/
    TV b{TV::zeros()};
    TV dbdt{TV::zeros()};
  };

  template <typename T, int dim> using GenericCollider
      = variant<Collider<AnalyticLevelSet<analytic_geometry_e::Plane, T, dim>>,
                Collider<AnalyticLevelSet<analytic_geometry_e::Cuboid, T, dim>>,
                Collider<AnalyticLevelSet<analytic_geometry_e::Sphere, T, dim>>,
                Collider<LevelSet<T, dim>>, Collider<HeightField<T>>>;

}  // namespace zs
