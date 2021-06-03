#pragma once
#include <type_traits>

#include "zensim/math/Vec.h"

namespace zs {

  // enum struct bv_e : char { aabb, obb, sphere, convex };
  template <typename Derived, typename T, int dim> struct BoundingVolumeInterface {
    static_assert(std::is_floating_point_v<T>, "T for bounding volume should be floating point!");

    using TV = vec<T, dim>;

    constexpr std::tuple<TV, TV> getBoundingBox() const noexcept {
      return selfPtr()->do_getBoundingBox();
    }
    constexpr TV getBoxCenter() const noexcept { return selfPtr()->do_getBoxCenter(); }

  protected:
    constexpr std::tuple<TV, TV> do_getBoundingBox() const noexcept {
      return std::make_tuple(TV::zeros(), TV::zeros());
    }
    constexpr TV do_getBoxCenter() const noexcept {
      auto &&[lo, hi] = getBoundingBox();
      return (lo + hi) / 2;
    }

    constexpr Derived *selfPtr() noexcept { return static_cast<Derived *>(this); }
    constexpr const Derived *selfPtr() const noexcept { return static_cast<const Derived *>(this); }
  };

}  // namespace zs