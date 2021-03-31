#pragma once

#include <type_traits>

namespace zs {

  /// pre C++14 impl, https://zh.cppreference.com/w/cpp/types/void_t
  /// check ill-formed types
  // template <typename... Ts> struct make_void { using type = void; };
  // template <typename... Ts> using void_t = typename make_void<Ts...>::type;
  template <typename... Ts> using void_t = std::void_t<Ts...>;

  /// SFINAE
  template <bool B> struct enable_if;
  template <> struct enable_if<true> { using type = char; };
  template <bool B> using enable_if_t = typename enable_if<B>::type;
  template <bool... Bs> using enable_if_all = typename enable_if<(Bs && ...)>::type;
  template <bool... Bs> using enable_if_any = typename enable_if<(Bs || ...)>::type;
  /// underlying_type
  /// common_type

}  // namespace zs
