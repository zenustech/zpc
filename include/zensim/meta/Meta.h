#pragma once

#include <initializer_list>
#include <type_traits>

#include "../TypeAlias.hpp"
#include "../ZpcMeta.hpp"

namespace zs {

  /// underlying_type
  /// common_type

  ///
  /// type decorato
  ///

  /// https://zh.cppreference.com/w/cpp/utility/tuple/make_tuple
  /// decay+unref
  template <class T> struct unwrap_refwrapper<std::reference_wrapper<T>> {
    using type = T &;
  };
  template <class T> struct is_refwrapper<std::reference_wrapper<T>> : std::true_type {};
  // specialization for std::integral_constant
  template <auto N> struct is_value_wrapper<std::integral_constant<decltype(N), N>> : true_type {};

  /// arithmetic type
  constexpr wrapt<u8> u8_c{};
  constexpr wrapt<int> int_c{};
  constexpr wrapt<uint> uint_c{};
  constexpr wrapt<i16> i16_c{};
  constexpr wrapt<i32> i32_c{};
  constexpr wrapt<i64> i64_c{};
  constexpr wrapt<u16> u16_c{};
  constexpr wrapt<u32> u32_c{};
  constexpr wrapt<u64> u64_c{};
  constexpr wrapt<f32> f32_c{};
  constexpr wrapt<float> float_c{};
  constexpr wrapt<f64> f64_c{};
  constexpr wrapt<double> double_c{};
  template <typename T> constexpr wrapt<enable_if_type<is_arithmetic_v<T>, T>> number_c{};

  ///
  /// detection
  ///
  namespace detail {
    template <class Default, class AlwaysVoid, template <class...> class Op, class... Args>
    struct detector {
      using value_t = false_type;
      using type = Default;
    };

    template <class Default, template <class...> class Op, class... Args>
    struct detector<Default, void_t<Op<Args...>>, Op, Args...> {
      using value_t = true_type;
      using type = Op<Args...>;
    };

  }  // namespace detail

  struct nonesuch {
    nonesuch() = delete;
    template <typename T> nonesuch(std::initializer_list<T>) = delete;
    ~nonesuch() = delete;
    nonesuch(const nonesuch &) = delete;
    void operator=(nonesuch const &) = delete;
  };

  template <template <class...> class Op, class... Args> using is_detected =
      typename detail::detector<nonesuch, void, Op, Args...>::value_t;

  template <template <class...> class Op, class... Args> using detected_t =
      typename detail::detector<nonesuch, void, Op, Args...>::type;

  template <typename Default, template <typename...> class Op, typename... Args> using detected_or
      = detail::detector<Default, void, Op, Args...>;

  template <typename Default, template <typename...> class Op, typename... Args> using detected_or_t
      = typename detected_or<Default, Op, Args...>::type;

}  // namespace zs
