#pragma once

#include <initializer_list>
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

  ///
  /// type decorato
  ///
  template <class T> struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
  };
  template <class T> using remove_cvref_t = typename remove_cvref<T>::type;

  /// vref
  template <class T> struct remove_vref {
    using type = std::remove_volatile_t<std::remove_reference_t<T>>;
  };
  template <class T> using remove_vref_t = typename remove_vref<T>::type;

  /// https://zh.cppreference.com/w/cpp/utility/tuple/make_tuple
  /// decay+unref
  template <class T> struct unwrap_refwrapper { using type = T; };
  template <class T> struct unwrap_refwrapper<std::reference_wrapper<T>> { using type = T &; };
  template <class T> using special_decay_t =
      typename unwrap_refwrapper<typename std::decay_t<T>>::type;

  ///
  /// fundamental type-value wrapper
  ///
  template <typename T> struct wrapt { using type = T; };
  /// wrap at most 1 layer
  template <typename T> struct wrapt<wrapt<T>> { using type = T; };
  template <auto N> using wrapv = std::integral_constant<decltype(N), N>;

  template <typename Tn, Tn N> using integral_v = std::integral_constant<Tn, N>;
  template <typename> struct is_integral_constant : std::false_type {};
  template <typename T, T v> struct is_integral_constant<integral_v<T, v>> : std::true_type {};
  template <std::size_t N> using index_v = std::integral_constant<std::size_t, N>;

  static constexpr std::true_type true_v{};
  static constexpr std::false_type false_v{};

  ///
  /// detection
  ///
  namespace detail {
    template <class Default, class AlwaysVoid, template <class...> class Op, class... Args>
    struct detector {
      using value_t = std::false_type;
      using type = Default;
    };

    template <class Default, template <class...> class Op, class... Args>
    struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
      using value_t = std::true_type;
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

  template <class Default, template <class...> class Op, class... Args> using detected_or
      = detail::detector<Default, void, Op, Args...>;

}  // namespace zs
