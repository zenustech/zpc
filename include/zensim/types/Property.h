#pragma once
#include <type_traits>

#include "zensim/meta/Meta.h"

namespace zs {

  enum struct attrib_e : unsigned char { scalar = 0, vec, mat, affine };
  constexpr auto scalar_v = wrapv<attrib_e::scalar>{};
  constexpr auto vec_v = wrapv<attrib_e::vec>{};
  constexpr auto mat_v = wrapv<attrib_e::mat>{};
  constexpr auto affine_v = wrapv<attrib_e::affine>{};

  enum struct layout_e : int { aos = 0, soa, aosoa };
  constexpr auto aos_v = wrapv<layout_e::aos>{};
  constexpr auto soa_v = wrapv<layout_e::soa>{};
  constexpr auto aosoa_v = wrapv<layout_e::aosoa>{};

  /// comparable
  template <typename T> struct is_equality_comparable {
  private:
    static void *conv(bool);
    template <typename U> static std::true_type test(
        decltype(conv(std::declval<U const &>() == std::declval<U const &>())),
        decltype(conv(!std::declval<U const &>() == std::declval<U const &>())));
    template <typename U> static std::false_type test(...);

  public:
    static constexpr bool value = decltype(test<T>(nullptr, nullptr))::value;
  };

}  // namespace zs
