#pragma once
#include "zensim/ZpcMeta.hpp"

namespace zs {

  template <typename T_> struct VectorViewLite {  // T may be const
    using value_type = remove_const_t<T_>;
    using size_type = size_t;

    static constexpr bool is_const_structure = is_const<T_>::value;

    constexpr VectorViewLite() noexcept = default;
    VectorViewLite(T_* const v) noexcept : _vector{v} {}

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr decltype(auto) operator[](size_type i) {
      return _vector[i];
    }
    constexpr auto operator[](size_type i) const { return _vector[i]; }

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr decltype(auto) operator()(size_type i) {
      return _vector[i];
    }
    constexpr auto operator()(size_type i) const { return _vector[i]; }

    T_* _vector{nullptr};
  };

}  // namespace zs