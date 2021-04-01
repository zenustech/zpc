#pragma once
#include "VdbLevelSet.h"
#include "zensim/math/Vec.h"
#include "zensim/types/Object.h"

namespace zs {

  template <typename Derived, typename T, int dim> struct LevelSetInterface
      : Inherit<Object, LevelSetInterface<Derived, T, dim>> {
    using TV = vec<T, dim>;
    constexpr T getSignedDistance(const TV &X) const noexcept {
      return self().getSignedDistance(X);
    }
    constexpr TV getNormal(const TV &X) const noexcept { return self().getNormal(X); }
    constexpr TV getMaterialVelocity(const TV &X) const noexcept {
      return self().getMaterialVelocity(X);
    }
    constexpr decltype(auto) getBoundingBox() const noexcept { return self().getBoundingBox(); }

  protected:
    constexpr auto &self() noexcept { return static_cast<Derived &>(*this); }
    constexpr const auto &self() const noexcept { return static_cast<const Derived &>(*this); }
  };

}  // namespace zs
