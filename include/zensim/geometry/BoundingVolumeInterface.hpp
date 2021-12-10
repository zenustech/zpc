#pragma once
#include <algorithm>
#include <type_traits>

#include "zensim/math/Vec.h"

namespace zs {

  // enum struct bv_e : char { aabb, obb, sphere, convex };
  template <typename Derived> struct BoundingVolumeInterface {
    constexpr auto getBoundingBox() const noexcept {
      return static_cast<const Derived *>(this)->do_getBoundingBox();
    }
    constexpr auto getBoxCenter() const noexcept {
      return static_cast<const Derived *>(this)->do_getBoxCenter();
    }
    constexpr auto getBoxSideLengths() const noexcept {
      return static_cast<const Derived *>(this)->do_getBoxSideLengths();
    }
    template <typename VecT, typename Self = Derived,
              enable_if_all<VecT::dim == 1, VecT::extent == Self::dim> = 0>
    constexpr auto getUniformCoord(const VecInterface<VecT> &pos) const noexcept {
      return static_cast<const Derived *>(this)->do_getUniformCoord(pos);
    }

  protected:
    template <typename Self> using TV = vec<typename Self::value_type, Self::dim>;
    template <typename Self = Derived, typename TV = TV<Self>>
    constexpr std::tuple<TV, TV> do_getBoundingBox() const noexcept {
      return std::make_tuple(TV::zeros(), TV::zeros());
    }
    template <typename Self = Derived, typename TV = TV<Self>>
    constexpr TV do_getBoxCenter() const noexcept {
      auto &&[lo, hi] = getBoundingBox();
      return (lo + hi) / 2;
    }
    template <typename Self = Derived, typename TV = TV<Self>>
    constexpr TV do_getBoxSideLengths() const noexcept {
      auto &&[lo, hi] = getBoundingBox();
      return hi - lo;
    }
    template <typename VecT, typename Self = Derived, typename TV = TV<Self>,
              enable_if_all<VecT::dim == 1, VecT::extent == Self::dim,
                            std::is_floating_point_v<typename Self::value_type>> = 0>
    constexpr TV do_getUniformCoord(const VecInterface<VecT> &pos) const noexcept {
      auto &&[lo, offset] = getBoundingBox();
      const auto lengths = offset - lo;
      offset = pos - lo;
      for (int d = 0; d != Self::dim; ++d)
        offset[d] = std::clamp(offset[d], (typename Self::value_type)0, lengths[d]) / lengths[d];
      return offset;
    }
  };

}  // namespace zs