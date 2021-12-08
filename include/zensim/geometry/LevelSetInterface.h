#pragma once
#include "BoundingVolumeInterface.hpp"

namespace zs {

  template <typename Derived, typename T, int dim> struct LevelSetInterface
      : BoundingVolumeInterface<Derived, T, dim> {
    using base_t = BoundingVolumeInterface<Derived, T, dim>;
    using TV = typename base_t::TV;

    constexpr T getSignedDistance(const TV &x) const noexcept {
      return selfPtr()->getSignedDistance(x);
    }
    constexpr TV getNormal(const TV &x) const noexcept { return selfPtr()->getNormal(x); }
    constexpr TV getMaterialVelocity(const TV &x) const noexcept {
      return selfPtr()->getMaterialVelocity(x);
    }
    using base_t::getBoundingBox;
    using base_t::getBoxCenter;
    using base_t::selfPtr;
  };

}  // namespace zs
