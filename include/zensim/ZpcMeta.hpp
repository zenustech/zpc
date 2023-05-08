#pragma once
/// can be used in py interop

namespace zs {

  ///
  /// SFINAE
  ///
  template <bool B> struct enable_if;
  template <> struct enable_if<true> {
    using type = int;
  };
  template <bool B> using enable_if_t = typename enable_if<B>::type;
  template <bool... Bs> using enable_if_all = typename enable_if<(Bs && ...)>::type;
  template <bool... Bs> using enable_if_any = typename enable_if<(Bs || ...)>::type;

  ///
  /// conditional
  ///
  template <bool B> struct conditional_impl {
    template <class T, class F> using type = T;
  };
  template <> struct conditional_impl<false> {
    template <class T, class F> using type = F;
  };
  template <bool B, class T, class F> using conditional_t =
      typename conditional_impl<B>::template type<T, F>;

  ///
  /// fundamental building blocks
  ///
  template <class T, T v> struct integral_constant {
    static constexpr T value = v;
    using value_type = T;
    using type = integral_constant;  // using injected-class-name
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }  // since c++14
  };

  template <bool B> using bool_constant = integral_constant<bool, B>;
  using true_type = bool_constant<true>;
  using false_type = bool_constant<false>;

  ///
  /// fundamental type-value wrapper
  ///
  template <typename T> struct wrapt {
    using type = T;
  };
  /// wrap at most 1 layer
  template <typename T> struct wrapt<wrapt<T>> {
    using type = T;
  };
  template <typename T> constexpr wrapt<T> wrapt_c{};

  template <typename T> struct is_type_wrapper : false_type {};
  template <typename T> struct is_type_wrapper<wrapt<T>> : true_type {};
  template <typename T> constexpr bool is_type_wrapper_v = is_type_wrapper<T>::value;

  /*
    template <typename Tn, Tn N> using integral_t = integral_constant<Tn, N>;
    template <typename> struct is_integral : false_type {};
    template <typename T, T v> struct is_integral<integral_t<T, v>> : true_type {};

    template <auto N> using wrapv = integral_t<decltype(N), N>;
    template <auto N> constexpr wrapv<N> wrapv_v{};

    template <typename T> struct is_value_wrapper : false_type {};
    template <auto N> struct is_value_wrapper<wrapv<N>> : true_type {};
    template <typename T> constexpr bool is_value_wrapper_v = is_value_wrapper<T>::value;

    template <std::size_t N> using index_t = integral_t<std::size_t, N>;
    template <std::size_t N> constexpr index_t<N> index_c{};
    */

  /// pred
  template <class T> struct is_const : false_type {};
  template <class T> struct is_const<const T> : true_type {};

}  // namespace zs