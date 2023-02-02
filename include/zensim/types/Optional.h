#pragma once
#include <optional>

namespace zs {

  template <typename T> using optional = std::optional<T>;
  using nullopt_t = std::nullopt_t;
  constexpr auto nullopt = std::nullopt;

  template <typename T> struct add_optional { using type = optional<T>; };
  template <typename T> struct add_optional<optional<T>> { using type = optional<T>; };

  template <typename T> using add_optional_t = typename add_optional<T>::type;

}  // namespace zs
