#pragma once
#include "zensim/geometry/BoundingVolumeInterface.hpp"

namespace zs {

  template <typename Derived> struct LevelSetInterface : BoundingVolumeInterface<Derived> {
    template <typename VecT, typename LS = Derived,
              enable_if_all<VecT::dim == 1, VecT::extent == LS::dim> = 0>
    constexpr auto getSignedDistance(const VecInterface<VecT> &x) const noexcept ->
        typename LS::value_type {
      return static_cast<const Derived *>(this)->do_getSignedDistance(x);
    }
    template <typename VecT, typename LS = Derived,
              enable_if_all<VecT::dim == 1, VecT::extent == LS::dim> = 0>
    constexpr auto getNormal(const VecInterface<VecT> &x) const noexcept {
      return static_cast<const Derived *>(this)->do_getNormal(x);
    }
    template <typename VecT, typename LS = Derived,
              enable_if_all<VecT::dim == 1, VecT::extent == LS::dim> = 0>
    constexpr auto getMaterialVelocity(const VecInterface<VecT> &x) const noexcept {
      return static_cast<const Derived *>(this)->do_getMaterialVelocity(x);
    }

  protected:
    template <typename VecT, typename LS = Derived,
              enable_if_all<VecT::dim == 1, VecT::extent == LS::dim> = 0>
    constexpr auto do_getSignedDistance(const VecInterface<VecT> &x) const noexcept {
      return detail::deduce_numeric_max<typename Derived::value_type>();
    }
    template <typename VecT, typename LS = Derived,
              enable_if_all<VecT::dim == 1, VecT::extent == LS::dim> = 0>
    constexpr auto do_getNormal(const VecInterface<VecT> &x) const noexcept {
      return VecInterface<VecT>::constant(0);
    }
    template <typename VecT, typename LS = Derived,
              enable_if_all<VecT::dim == 1, VecT::extent == LS::dim> = 0>
    constexpr auto do_getMaterialVelocity(const VecInterface<VecT> &x) const noexcept {
      return VecInterface<VecT>::constant(0);
    }
  };

}  // namespace zs
