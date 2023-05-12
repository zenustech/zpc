#pragma once

#include <type_traits>
#include <utility>

#include "../TypeAlias.hpp"
#include "Functional.h"
#include "Meta.h"

namespace zs {

  namespace type_impl {
    template <typename T, T... Is, typename... Ts>
    struct indexed_types<std::integer_sequence<T, Is...>, Ts...> : indexed_type<Is, Ts>... {};
  }  // namespace type_impl

  /// type_seq
  

  template <typename TT, typename T> struct is_assignable {
    template <typename A, typename B, typename = void> struct pred {
      static constexpr bool value = false;
    };
    template <typename A, typename B>
    struct pred<A, B, void_t<decltype(declval<A>() = declval<B>())>> {
      static constexpr bool value = true;
    };
    template <typename... Ts, typename... Vs>
    static constexpr bool test(type_seq<Ts...>, type_seq<Vs...>) {
      if constexpr (sizeof...(Vs) == sizeof...(Ts))
        /// @note (std::is_assignable<Ts, Vs>::value && ...) may cause nvcc compiler error
        return (pred<Ts, Vs>::value && ...);
      else
        return false;
    }
    template <typename UU, typename U> static constexpr auto test(char) {
      if constexpr (is_type_seq_v<UU> && is_type_seq_v<U>)
        return integral<bool, test(UU{}, U{})>{};
      else
        return integral<bool, pred<UU, U>::value>{};
    }

  public:
    static constexpr bool value = decltype(test<TT, T>(0))::value;
  };
  template <typename TT, typename T> constexpr bool is_assignable_v = is_assignable<TT, T>::value;

  /** placeholder */
  namespace index_literals {
    // ref: numeric UDL
    // Embracing User Defined Literals Safely for Types that Behave as though Built-in
    // Pablo Halpern
    template <auto partial> constexpr auto index_impl() noexcept { return partial; }
    template <auto partial, char c0, char... c> constexpr auto index_impl() noexcept {
      if constexpr (c0 == '\'')
        return index_impl<partial, c...>();
      else {
        using Tn = decltype(partial);
        static_assert(c0 >= '0' && c0 <= '9', "Invalid non-numeric character");
        static_assert(partial < (limits<Tn>::max() - (c0 - '0')) / 10 + 1,
                      "numeric literal overflow");
        return index_impl<partial *(Tn)10 + (Tn)(c0 - '0'), c...>();
      }
    }

    template <char... c> constexpr auto operator""_th() noexcept {
      constexpr auto id = index_impl<(size_t)0, c...>();
      return index_c<id>;
    }
  }  // namespace index_literals

  template <char... c> constexpr auto operator""_c() noexcept {
    return index_literals::operator""_th<c...>();
  }

  template <size_t... Ns> constexpr value_seq<Ns...> dim_c{};

  namespace placeholders {
    using placeholder_type = size_t;
    constexpr auto _0 = integral<placeholder_type, 0>{};
    constexpr auto _1 = integral<placeholder_type, 1>{};
    constexpr auto _2 = integral<placeholder_type, 2>{};
    constexpr auto _3 = integral<placeholder_type, 3>{};
    constexpr auto _4 = integral<placeholder_type, 4>{};
    constexpr auto _5 = integral<placeholder_type, 5>{};
    constexpr auto _6 = integral<placeholder_type, 6>{};
    constexpr auto _7 = integral<placeholder_type, 7>{};
    constexpr auto _8 = integral<placeholder_type, 8>{};
    constexpr auto _9 = integral<placeholder_type, 9>{};
    constexpr auto _10 = integral<placeholder_type, 10>{};
    constexpr auto _11 = integral<placeholder_type, 11>{};
    constexpr auto _12 = integral<placeholder_type, 12>{};
    constexpr auto _13 = integral<placeholder_type, 13>{};
    constexpr auto _14 = integral<placeholder_type, 14>{};
    constexpr auto _15 = integral<placeholder_type, 15>{};
    constexpr auto _16 = integral<placeholder_type, 16>{};
    constexpr auto _17 = integral<placeholder_type, 17>{};
    constexpr auto _18 = integral<placeholder_type, 18>{};
    constexpr auto _19 = integral<placeholder_type, 19>{};
    constexpr auto _20 = integral<placeholder_type, 20>{};
    constexpr auto _21 = integral<placeholder_type, 21>{};
    constexpr auto _22 = integral<placeholder_type, 22>{};
    constexpr auto _23 = integral<placeholder_type, 23>{};
    constexpr auto _24 = integral<placeholder_type, 24>{};
    constexpr auto _25 = integral<placeholder_type, 25>{};
    constexpr auto _26 = integral<placeholder_type, 26>{};
    constexpr auto _27 = integral<placeholder_type, 27>{};
    constexpr auto _28 = integral<placeholder_type, 28>{};
    constexpr auto _29 = integral<placeholder_type, 29>{};
    constexpr auto _30 = integral<placeholder_type, 30>{};
    constexpr auto _31 = integral<placeholder_type, 31>{};
  }  // namespace placeholders
  using place_id = placeholders::placeholder_type;

}  // namespace zs
