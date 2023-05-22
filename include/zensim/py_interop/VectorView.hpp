#pragma once
#include "zensim/ZpcMeta.hpp"

namespace zs {

  template <typename T> struct VectorViewLite {  // T may be const
    using size_type = size_t;

    static constexpr bool is_const_structure = is_const<T>::value;

    VectorViewLite() noexcept = default;
    VectorViewLite(T* const v) noexcept : _vector{v} {}

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr decltype(auto) operator[](size_type i) {
      return _vector[i];
    }
    constexpr auto operator[](size_type i) const { return _vector[i]; }

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr decltype(auto) operator()(size_type i) {
      return _vector[i];
    }
    constexpr decltype(auto) operator()(size_type i) const { return _vector[i]; }

    T* _vector{nullptr};
  };

}  // namespace zs