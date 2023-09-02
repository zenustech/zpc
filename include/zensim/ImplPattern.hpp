#pragma once
#include "zensim/ZpcMeta.hpp"

namespace zs {

  template <template <typename> class... Skills> struct Object
      : private Skills<Object<Skills...>>... {};
  template <typename Derived, template <typename> class... Skills> struct Mixin
      : public Skills<Derived>... {};

  template <typename Derived> struct Visitee {
    constexpr auto derivedPtr() noexcept { return static_cast<Derived*>(this); }
    constexpr auto derivedPtr() const noexcept { return static_cast<const Derived*>(this); }
    constexpr auto derivedPtr() volatile noexcept { return static_cast<volatile Derived*>(this); }
    constexpr auto derivedPtr() const volatile noexcept {
      return static_cast<const volatile Derived*>(this);
    }

    constexpr void accept(...) {}
    constexpr void accept(...) const {}

    template <typename Visitor> constexpr auto accept(Visitor&& visitor)
        -> decltype(FWD(visitor)(declval<Derived&>())) {
      return FWD(visitor)(*derivedPtr());
    }
    template <typename Visitor> constexpr auto accept(Visitor&& visitor) const
        -> decltype(FWD(visitor)(declval<const Derived&>())) {
      return FWD(visitor)(*derivedPtr());
    }
    template <typename Policy, typename Visitor>
    constexpr auto accept(Policy&& pol, Visitor&& visitor)
        -> decltype(FWD(visitor)(FWD(pol), declval<Derived&>())) {
      return FWD(visitor)(FWD(pol), *derivedPtr());
    }
    template <typename Policy, typename Visitor>
    constexpr auto accept(Policy&& pol, Visitor&& visitor) const
        -> decltype(FWD(visitor)(FWD(pol), declval<const Derived&>())) {
      return FWD(visitor)(FWD(pol), *derivedPtr());
    }
  };

}  // namespace zs