#pragma once
#include "zensim/ZpcMeta.hpp"

namespace zs {

  template <typename T_> struct DenseFieldViewLite {  // T may be const
    using value_type = remove_const_t<T_>;
    using size_type = size_t;

    static constexpr bool is_const_structure = is_const<T_>::value;

    constexpr DenseFieldViewLite() noexcept = default;
    DenseFieldViewLite(T_* const v, const size_type* const bases, size_type dim) noexcept
        : _field{v}, _bases{bases}, _dim{dim} {}

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr decltype(auto) operator[](size_type i) {
      return _field[i];
    }
    constexpr auto operator[](size_type i) const { return _field[i]; }

    template <typename... Args, enable_if_all<is_integral_v<Args>...> = 0>
    constexpr size_type linearOffset(Args... is) const noexcept {
      size_type offset = 0, i = 0;
      ((void)(offset += (size_type)is * _bases[++i]), ...);
      return offset;
    }
    template <typename... Args, bool V = !is_const_structure && (is_integral_v<Args> && ...),
              enable_if_t<V> = 0>
    constexpr decltype(auto) operator()(Args... is) {
      const size_type offset = linearOffset(zs::move(is)...);
      return operator[](offset);
    }
    template <typename... Args, enable_if_all<is_integral_v<Args>...> = 0>
    constexpr auto operator()(Args... is) const {
      const size_type offset = linearOffset(zs::move(is)...);
      return operator[](offset);
    }

    T_* _field{nullptr};
    const size_type* _bases{nullptr};  // "_dim + 1" elements, last one being 1
    size_type _dim{0};
  };

}  // namespace zs