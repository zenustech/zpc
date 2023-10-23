#pragma once
/// can be used in py interop

namespace zs {

  // ref: https://en.cppreference.com/w/cpp/meta
  // https://stackoverflow.com/questions/20181702/which-type-traits-cannot-be-implemented-without-compiler-hooks

  // ref: https://stackoverflow.com/questions/1119370/where-do-i-find-the-definition-of-size-t
  using size_t = decltype(sizeof(int));
  using nullptr_t = decltype(nullptr);

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

  // conditional
  template <bool B> struct conditional_impl {
    template <class T, class F> using type = T;
  };
  template <> struct conditional_impl<false> {
    template <class T, class F> using type = F;
  };
  template <bool B, class T, class F> using conditional_t =
      typename conditional_impl<B>::template type<T, F>;
  template <bool B, typename T = void> using enable_if_type = conditional_t<B, T, enable_if_t<B>>;

  using sint_t
      = conditional_t<sizeof(long long int) == sizeof(size_t), long long int,
                      conditional_t<sizeof(long int) == sizeof(size_t), long int,
                                    conditional_t<sizeof(int) == sizeof(size_t), int, short>>>;
  static_assert(alignof(sint_t) == alignof(size_t) && sizeof(sint_t) == sizeof(size_t),
                "sint_t not properly deduced!");

  ///
  /// fundamental building blocks
  ///
  /// @note fundamental type-value-seq wrapper
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

  template <typename Tn, Tn N> using integral = integral_constant<Tn, N>;
  template <typename> struct is_integral_constant : false_type {};
  template <typename T, T v> struct is_integral_constant<integral<T, v>> : true_type {};
  constexpr true_type true_c{};
  constexpr false_type false_c{};

  template <class...> using void_t = void;
  struct failure_type {};  // no [type] member, should never be used publically

  template <typename T> struct wrapt {
    using type = T;
  };
  template <typename T> struct wrapt<wrapt<T>> {
    // wrap at most 1 layer
    using type = T;
  };
  template <typename T> constexpr wrapt<T> wrapt_c{};
  template <auto N> using wrapv = integral_constant<decltype(N), N>;
  template <auto N> constexpr wrapv<N> wrapv_c{};

  template <typename T> struct is_type_wrapper : false_type {};
  template <typename T> struct is_type_wrapper<wrapt<T>> : true_type {};
  template <typename T> constexpr bool is_type_wrapper_v = is_type_wrapper<T>::value;

  template <typename T> struct is_value_wrapper : false_type {};
  template <auto N> struct is_value_wrapper<wrapv<N>> : true_type {};
  template <typename T> constexpr bool is_value_wrapper_v = is_value_wrapper<T>::value;

  template <zs::size_t N> using index_t = integral_constant<zs::size_t, N>;
  template <zs::size_t N> constexpr index_t<N> index_c{};

  template <class T, T... Is> class integer_sequence {
    using value_type = T;
    static constexpr size_t size() noexcept { return sizeof...(Is); }
  };
  template <size_t... Is> using index_sequence = integer_sequence<zs::size_t, Is...>;
  namespace detail {
    /// @note the following impl is also intriguing, less funcs analyzed but more verbose AST
    /// https://github.com/taocpp/sequences/blob/main/include/tao/seq/make_integer_sequence.hpp
    template <bool Odd, typename T, T... Is>
    constexpr auto concat_sequence(integer_sequence<T, Is...>) {
      if constexpr (Odd)
        return integer_sequence<T, Is..., (sizeof...(Is) + Is)..., (sizeof...(Is) * 2)>{};
      else
        return integer_sequence<T, Is..., (sizeof...(Is) + Is)...>{};
    }
    template <typename T, size_t N, enable_if_t<(N <= 1)> = 0> constexpr auto gen_integer_seq() {
      if constexpr (N == 0)
        return integer_sequence<T>{};
      else  // if constexpr (N == 1)
        return integer_sequence<T, 0>{};
    }
    template <typename T, size_t N, enable_if_t<(N > 1)> = 0> constexpr auto gen_integer_seq() {
      return concat_sequence<N & 1>(gen_integer_seq<T, N / 2>());
    }
  }  // namespace detail
  template <typename T, T N> using make_integer_sequence
      = decltype(detail::gen_integer_seq<T, N>());
  template <zs::size_t N> using make_index_sequence = make_integer_sequence<zs::size_t, N>;
  template <class... T> using index_sequence_for = make_index_sequence<sizeof...(T)>;

  /// indexable type list to avoid recursion
  namespace type_impl {
    template <auto I, typename T> struct indexed_type {
      using type = T;
      static constexpr auto value = I;
    };
    template <typename, typename... Ts> struct indexed_types;

    template <typename T, T... Is, typename... Ts>
    struct indexed_types<integer_sequence<T, Is...>, Ts...> : indexed_type<Is, Ts>... {};
    template <typename T> struct indexed_types<integer_sequence<T>> : wrapt<T> {};

    // use pointer rather than reference as in taocpp! [incomplete type error]
    template <auto I, typename T> static indexed_type<I, T> extract_type(indexed_type<I, T> *);
    template <auto I, typename Ti>
    static wrapt<failure_type> extract_type(wrapt<Ti> *);  // indicates no type
    template <typename T, auto I> static indexed_type<I, T> extract_index(indexed_type<I, T> *);
    template <typename T, typename Ti> static wrapv<(~(size_t)0)> extract_index(wrapt<Ti> *);
  }  // namespace type_impl

  template <typename... Ts> struct type_seq;
  template <auto... Ns> struct value_seq;

  template <typename> struct is_type_seq : false_type {};
  template <typename... Ts> struct is_type_seq<type_seq<Ts...>> : true_type {};
  template <typename SeqT> static constexpr bool is_type_seq_v = is_type_seq<SeqT>::value;

  /// generate index sequence declaration
  template <typename> struct gen_seq_impl;
  template <size_t... Is> struct gen_seq_impl<index_sequence<Is...>> {
    /// arithmetic sequences
    template <auto N0 = 0, auto Step = 1> using arithmetic = index_sequence<static_cast<size_t>(
        static_cast<signed long long int>(N0)
        + static_cast<signed long long int>(Is) * static_cast<signed long long int>(Step))...>;
    using ascend = arithmetic<0, 1>;
    using descend = arithmetic<sizeof...(Is) - 1, -1>;
    template <auto J> using uniform = integer_sequence<decltype(J), ((void)Is, J)...>;
    template <auto J> using constant = integer_sequence<decltype(J), ((void)Is, J)...>;
    template <auto J> using uniform_vseq = value_seq<((void)Is, J)...>;
    /// types with uniform type/value params
    template <template <typename...> class T, typename Arg> using uniform_types_t
        = T<enable_if_type<(Is >= 0), Arg>...>;
    template <template <typename...> class T, typename Arg>
    static constexpr auto uniform_values(const Arg &arg) {
      return uniform_types_t<T, Arg>{((void)Is, arg)...};
    }
    template <template <auto...> class T, auto Arg> using uniform_values_t
        = T<(Is >= 0 ? Arg : 0)...>;
  };
  template <zs::size_t N> using gen_seq = gen_seq_impl<make_index_sequence<N>>;

  template <typename... Seqs> struct concat;
  template <typename... Seqs> using concat_t = typename concat<Seqs...>::type;

  ///
  /// type predicates
  ///
  // is_same
  struct is_same_wrapper_base {
    static constexpr bool is_same(void *) { return false; };
  };
  template <typename T> struct wrapper : is_same_wrapper_base {
    using is_same_wrapper_base::is_same;
    static constexpr bool is_same(wrapper<T> *) { return true; };
  };
  template <typename T1, typename T2> using is_same
      = bool_constant<wrapper<T1>::is_same((wrapper<T2> *)nullptr)>;
  template <typename T1, typename T2> constexpr auto is_same_v = is_same<T1, T2>::value;
  // rank
  template <typename> struct rank : integral_constant<size_t, 0> {};
  template <typename T, size_t S> struct rank<T[S]>
      : integral_constant<size_t, 1 + rank<T>::value> {};
  template <typename T> struct rank<T[]> : integral_constant<size_t, 1 + rank<T>::value> {};
  // extent (of a certain dimension, from left to right indexed [0, 1, ..., n - 1])
  template <typename, unsigned dim = 0> struct extent : integral_constant<size_t, 0> {};
  template <typename T, unsigned dim, size_t S> struct extent<T[S], dim>
      : integral_constant<size_t, dim == 0 ? S : extent<T, dim - 1>::value> {};
  template <typename T, unsigned dim> struct extent<T[], dim>
      : integral_constant<size_t, dim == 0 ? 0 : extent<T, dim - 1>::value> {};
  // (C) array
  template <class T> struct is_array : false_type {};
  template <class T> struct is_array<T[]> : true_type {};
  template <class T, size_t N> struct is_array<T[N]> : true_type {};
  template <class T> constexpr bool is_array_v = is_array<T>::value;
  template <class T> struct is_unbounded_array : false_type {};
  template <class T> struct is_unbounded_array<T[]> : true_type {};
  template <class T> constexpr bool is_unbounded_array_v = is_unbounded_array<T>::value;
  template <class T> constexpr bool is_array_unknown_bounds_v = is_array_v<T> && !extent<T>::value;
  // const
  template <class T> struct is_const : false_type {};
  template <class T> struct is_const<const T> : true_type {};
  template <class T> constexpr bool is_const_v = is_const<T>::value;
  // volatile
  template <class T> struct is_volatile : false_type {};
  template <class T> struct is_volatile<volatile T> : true_type {};
  template <class T> constexpr bool is_volatile_v = is_volatile<T>::value;
  // reference
  template <class T> struct is_lvalue_reference : false_type {};
  template <class T> struct is_lvalue_reference<T &> : true_type {};
  template <class T> constexpr bool is_lvalue_reference_v = is_lvalue_reference<T>::value;
  template <class T> struct is_rvalue_reference : false_type {};
  template <class T> struct is_rvalue_reference<T &&> : true_type {};
  template <class T> constexpr bool is_rvalue_reference_v = is_rvalue_reference<T>::value;
  template <class T> struct is_reference : false_type {};
  template <class T> struct is_reference<T &> : true_type {};
  template <class T> struct is_reference<T &&> : true_type {};
  template <class T> constexpr bool is_reference_v = is_reference<T>::value;

  template <class T, typename = void> struct is_referenceable : false_type {};
  template <class T> struct is_referenceable<T, void_t<T &>> : true_type {};
  template <class T> constexpr bool is_referenceable_v = is_referenceable<T>::value;
  // function
  /// @note
  /// https://github.com/Quuxplusone/from-scratch/blob/master/include/scratch/bits/type-traits/compiler-magic.md
  // is_enum
  // ref: https://stackoverflow.com/questions/11316912/is-enum-implementation
  template <class T> struct is_enum : bool_constant<__is_enum(T)> {};
  template <class T> constexpr bool is_enum_v = is_enum<T>::value;
  template <class T> struct is_function
      : integral_constant<bool, !is_const<const T>::value && !is_reference<T>::value> {};
  template <class T> constexpr bool is_function_v = is_function<T>::value;
  // is_trivial
  template <class T> struct is_trivial : bool_constant<__is_trivial(T)> {};
  template <class T> constexpr bool is_trivial_v = is_trivial<T>::value;
  // is_trivially_constructible
  template <class T, typename... Args> struct is_trivially_constructible
      : bool_constant<__is_trivially_constructible(T, Args...)> {};
  // is_trivially_copyable
  template <class T> struct is_trivially_copyable : bool_constant<__is_trivially_copyable(T)> {};
  template <class T> constexpr bool is_trivially_copyable_v = is_trivially_copyable<T>::value;
  // is_trivially_assignable
  template <typename T, typename U> struct is_trivially_assignable
      : bool_constant<__is_trivially_assignable(T, U)> {};
  // is_trivially_destructible
  // has_virtual_destructor
  template <typename T> struct has_virtual_destructor : bool_constant<__has_virtual_destructor(T)> {
  };
  // is_standard_layout
  template <class T> struct is_standard_layout : public bool_constant<__is_standard_layout(T)> {};
  template <class T> constexpr bool is_standard_layout_v = is_standard_layout<T>::value;
  // is_empty
  template <class T> struct is_empty : public bool_constant<__is_empty(T)> {};
  template <class T> constexpr bool is_empty_v = is_empty<T>::value;
  // is_polymorphic
  template <class T> struct is_polymorphic : public bool_constant<__is_polymorphic(T)> {};
  template <class T> constexpr bool is_polymorphic_v = is_polymorphic<T>::value;
  // is_final
  template <class T> struct is_final : public bool_constant<__is_final(T)> {};
  template <class T> constexpr bool is_final_v = is_final<T>::value;
  // is_abstract
  template <class T> struct is_abstract : public bool_constant<__is_abstract(T)> {};
  template <class T> constexpr bool is_abstract_v = is_abstract<T>::value;

  ///
  /// type decoration
  ///
  // remove_reference
  template <class T> struct remove_reference {
    using type = T;
  };
  template <class T> struct remove_reference<T &> {
    using type = T;
  };
  template <class T> struct remove_reference<T &&> {
    using type = T;
  };
  template <typename T> using remove_reference_t = typename remove_reference<T>::type;
#define RM_REF_T(...) ::zs::remove_reference_t<decltype(__VA_ARGS__)>
#define ZS_TYPE(...) typename ::zs::remove_reference_t<decltype(__VA_ARGS__)>::type
#define ZS_VALUE(...) typename ::zs::remove_reference_t<decltype(__VA_ARGS__)>::value
  // remove cv
  template <class T> struct remove_cv {
    using type = T;
  };
  template <class T> struct remove_cv<const T> {
    using type = T;
  };
  template <class T> struct remove_cv<volatile T> {
    using type = T;
  };
  template <class T> struct remove_cv<const volatile T> {
    using type = T;
  };
  template <typename T> using remove_cv_t = typename remove_cv<T>::type;
  template <class T> struct remove_const {
    using type = T;
  };
  template <class T> struct remove_const<const T> {
    using type = T;
  };
  template <typename T> using remove_const_t = typename remove_const<T>::type;
  template <class T> struct remove_volatile {
    using type = T;
  };
  template <class T> struct remove_volatile<volatile T> {
    using type = T;
  };
  template <typename T> using remove_volatile_t = typename remove_volatile<T>::type;
  template <class T> struct remove_cvref {
    using type = typename remove_cv<typename remove_reference<T>::type>::type;
  };
  template <class T> using remove_cvref_t = typename remove_cvref<T>::type;
  template <class T> struct remove_vref {
    using type = typename remove_volatile<typename remove_reference<T>::type>::type;
  };
  template <class T> using remove_vref_t = typename remove_vref<T>::type;

#define RM_CVREF_T(...) ::zs::remove_cvref_t<decltype(__VA_ARGS__)>
  // #define RM_CVREF_T(...) ::std::remove_cvref_t<decltype(__VA_ARGS__)>

  template <class T> struct remove_pointer {
    using type = T;
  };
  template <class T> struct remove_pointer<T *> {
    using type = T;
  };
  template <class T> struct remove_pointer<T *const> {
    using type = T;
  };
  template <class T> struct remove_pointer<T *volatile> {
    using type = T;
  };
  template <class T> struct remove_pointer<T *const volatile> {
    using type = T;
  };
  template <typename T> using remove_pointer_t = typename remove_pointer<T>::type;

  // add_pointer
  namespace detail {
    template <class T> auto try_add_pointer(int) -> wrapt<remove_reference_t<T> *>;
    template <class T> auto try_add_pointer(...) -> wrapt<T>;

  }  // namespace detail
  template <class T> struct add_pointer : decltype(detail::try_add_pointer<T>(0)) {};
  template <typename T> using add_pointer_t = typename add_pointer<T>::type;
  // add reference
  namespace detail {
    template <class T>  // Note that `cv void&` is a substitution failure
    auto try_add_lvalue_reference(int) -> wrapt<T &>;
    template <class T>  // Handle T = cv void case
    auto try_add_lvalue_reference(...) -> wrapt<T>;

    template <class T> auto try_add_rvalue_reference(int) -> wrapt<T &&>;
    template <class T> auto try_add_rvalue_reference(...) -> wrapt<T>;
  }  // namespace detail
  template <class T> struct add_lvalue_reference
      : decltype(detail::try_add_lvalue_reference<T>(0)) {};
  template <typename T> using add_lvalue_reference_t = typename add_lvalue_reference<T>::type;
  template <class T> struct add_rvalue_reference
      : decltype(detail::try_add_rvalue_reference<T>(0)) {};
  template <typename T> using add_rvalue_reference_t = typename add_rvalue_reference<T>::type;
  // add_cv/const/volatile
  template <class T> struct add_cv {
    typedef const volatile T type;
  };
  template <class T> using add_cv_t = typename add_cv<T>::type;
  template <class T> struct add_const {
    typedef const T type;
  };
  template <class T> using add_const_t = typename add_const<T>::type;
  template <class T> struct add_volatile {
    typedef volatile T type;
  };
  template <class T> using add_volatile_t = typename add_volatile<T>::type;
  // remove_extent
  template <class T> struct remove_extent {
    using type = T;
  };
  template <class T> struct remove_extent<T[]> {
    using type = T;
  };
  template <class T, size_t N> struct remove_extent<T[N]> {
    using type = T;
  };
  template <class T> struct remove_all_extents {
    using type = T;
  };
  template <class T> struct remove_all_extents<T[]> {
    using type = typename remove_all_extents<T>::type;
  };
  template <class T, size_t N> struct remove_all_extents<T[N]> {
    using type = typename remove_all_extents<T>::type;
  };
  template <typename T> using remove_extent_t = typename remove_extent<T>::type;
  template <typename T> using remove_all_extents_t = typename remove_all_extents<T>::type;
  // underlying_type
  template <class T, bool = is_enum_v<T>> struct underlying_type {
    using type = __underlying_type(T);
  };
  template <class T> struct underlying_type<T, false> {};
  template <class T> using underlying_type_t = typename underlying_type<T>::type;
  /// decay
  template <class T> struct decay {
  private:
    using U = typename remove_reference<T>::type;

  public:
    using type = conditional_t<is_array<U>::value,
                               typename add_pointer<typename remove_extent<U>::type>::type,
                               conditional_t<is_function<U>::value, typename add_pointer<U>::type,
                                             typename remove_cv<U>::type>>;
  };
  template <typename T> using decay_t = typename decay<T>::type;

  ///
  /// useful constructs
  ///
  // declval
  template <typename T> constexpr bool always_false = false;
  template <typename T> typename add_rvalue_reference<T>::type declval() noexcept {
    static_assert(always_false<T>, "declval not allowed in an evaluated context");
  }
#define INST_(T) declval<T>()
#define DECL_(T) declval<T>()
  // forward
  /// https://stackoverflow.com/questions/27501400/the-implementation-of-stdforward
  template <class T> constexpr T &&forward(typename remove_reference<T>::type &t) noexcept {
    return static_cast<T &&>(t);
  }
  template <class T> constexpr T &&forward(typename remove_reference<T>::type &&t) noexcept {
    static_assert(!is_lvalue_reference<T>::value, "Can not forward an rvalue as an lvalue.");
    return static_cast<T &&>(t);
  }
  template <class T> constexpr remove_reference_t<T> &&move(T &&t) noexcept {
    return static_cast<remove_reference_t<T> &&>(t);
  }
/// https://vittorioromeo.info/index/blog/capturing_perfectly_forwarded_objects_in_lambdas.html
#define FWD(...) ::zs::forward<decltype(__VA_ARGS__)>(__VA_ARGS__)
  // is_valid
  /// ref: boost-hana
  /// boost/hana/type.hpp
  // copyright Louis Dionne 2013-2017
  // Distributed under the Boost Software License, Version 1.0.
  // (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)
  namespace detail {
    template <typename F, typename... Args> constexpr auto is_valid_impl(int) noexcept
        -> decltype(declval<F &&>()(declval<Args &&>()...), true_type{}) {
      return true_c;
    }
    template <typename F, typename... Args> constexpr auto is_valid_impl(...) noexcept {
      return false_c;
    }
    template <typename F> struct is_valid_fun {
      template <typename... Args> constexpr auto operator()(Args &&...) const noexcept {
        return is_valid_impl<F, Args &&...>(0);
      }
    };
  }  // namespace detail
  struct is_valid_t {
    template <typename F> constexpr auto operator()(F &&) const noexcept {
      return detail::is_valid_fun<F &&>{};
    }
    template <typename F, typename... Args> constexpr auto operator()(F &&, Args &&...) const {
      return detail::is_valid_impl<F &&, Args &&...>(0);
    }
  };
  constexpr is_valid_t is_valid{};
  // type_seq (impl)
  template <typename... Ts> struct type_seq {
    static constexpr bool is_value_sequence() noexcept {
      if constexpr (sizeof...(Ts) == 0)
        return true;
      else
        return (is_value_wrapper_v<Ts> && ...);
    }
    static constexpr bool all_values = is_value_sequence();

    using indices = index_sequence_for<Ts...>;

    static constexpr auto count = sizeof...(Ts);

    // type
    template <zs::size_t I> using type = typename decltype(type_impl::extract_type<I>(
        declval<add_pointer_t<type_impl::indexed_types<indices, Ts...>>>()))::type;

    /// @note when Ts=<>, type<0> is 'wrapv<0>'
    template <zs::size_t N = sizeof...(Ts), enable_if_t<(N == 1)> = 0>
    constexpr operator wrapt<type<0>>() const noexcept {
      return {};
    }

    // index
    template <typename T> static constexpr size_t count_occurencies() noexcept {
      if constexpr (sizeof...(Ts) == 0)
        return 0;
      else
        return (static_cast<size_t>(is_same_v<T, Ts>) + ...);
    }
    template <typename T> using occurencies_t = wrapv<count_occurencies<T>()>;
#if 0
    template <typename, typename = void> struct locator {
      using index = index_t<~(size_t)0>;
    };
    template <typename T> struct locator<T, enable_if_type<count_occurencies<T>() == 1>> {
      using index = integral<
          size_t, decltype(type_impl::extract_index<T>(
                      declval<add_pointer_t<type_impl::indexed_types<indices, Ts...>>>()))::value>;
    };
#else
    template <typename T> struct locator {
      using index = integral<
          size_t, decltype(type_impl::extract_index<T>(
                      declval<add_pointer_t<type_impl::indexed_types<indices, Ts...>>>()))::value>;
    };
#endif
    template <typename T> using index = typename locator<T>::index;

    // functor
    template <template <typename...> class T> using functor = T<Ts...>;

    ///
    /// operations
    ///
    template <auto I = 0> constexpr auto get_type(wrapv<I> = {}) const noexcept {
      return wrapt<type<I>>{};
    }
    template <typename T = void> constexpr auto get_index(wrapt<T> = {}) const noexcept {
      return index<T>{};
    }
    template <typename Ti, Ti... Is>
    constexpr auto filter(integer_sequence<Ti, Is...>) const noexcept {  // for tuple_cat
      return value_seq<index<type_seq<integral<Ti, 1>, integral<Ti, Is>>>::value...>{};
    }
    template <typename... Args> constexpr auto pair(type_seq<Args...>) const noexcept;
    template <typename Ti, Ti... Is>
    constexpr auto shuffle(integer_sequence<Ti, Is...>) const noexcept {
      if constexpr (all_values)
        return value_seq<type<Is>::value...>{};
      else
        return type_seq<type<Is>...>{};
    }
    template <typename Ti, Ti... Is>
    constexpr auto shuffle_join(integer_sequence<Ti, Is...>) const noexcept {
      static_assert((is_type_seq_v<Ts> && ...), "");
      return type_seq<typename Ts::template type<Is>...>{};
    }
  };
  ///

  /// select type by index
  template <zs::size_t I, typename TypeSeq> using select_type = typename TypeSeq::template type<I>;
  template <zs::size_t I, typename... Ts> using select_indexed_type
      = select_type<I, type_seq<Ts...>>;

  ///
  /// advanced predicates
  ///
  // void
  template <class T> struct is_void : is_same<void, remove_cv_t<T>> {};
  template <class T> constexpr bool is_void_v = is_same_v<void, remove_cv_t<T>>;
  // arithmetic
  namespace detail {
#if 0
    template <typename T> static auto test_integral(T t, T *p, void (*f)(T))
        -> decltype(reinterpret_cast<T>(t), f(0), p + t, true_type{});
#else
    template <typename T,
              typename = enable_if_t<!is_void_v<T> && !is_const_v<T> && !is_volatile_v<T>
                                     && !is_reference_v<T> && !__is_enum(T) && !__is_class(T)>>
    static auto test_integral(T t) -> decltype((char *)(nullptr) + t, true_type{});
#endif
    static false_type test_integral(...) noexcept;
  }  // namespace detail
#if 1
  template <typename T, typename = void> struct is_integral : false_type {};
  template <typename T> struct is_integral<
      T,
      void_t<T *, enable_if_all<is_same_v<decltype(detail::test_integral(declval<T>())), true_type>,
                                !is_const_v<T>, !is_volatile_v<T>, !__is_enum(T), !__is_class(T)>>>
      : true_type {};
  template <class T> constexpr bool is_integral_v
      = decltype(detail::test_integral(declval<T>()))::value;
  static_assert(is_integral<int>::value, "???");
  static_assert(!is_integral<float>::value, "???");
  static_assert(!is_integral<int &>::value, "???");
  static_assert(!is_integral<volatile int>::value, "???");
#else
  template <class T> struct is_integral : false_type {};
  template <> struct is_integral<bool> : true_type {};
  template <> struct is_integral<char> : true_type {};
  template <> struct is_integral<signed char> : true_type {};
  template <> struct is_integral<unsigned char> : true_type {};
  template <> struct is_integral<short> : true_type {};
  template <> struct is_integral<unsigned short> : true_type {};
  template <> struct is_integral<int> : true_type {};
  template <> struct is_integral<unsigned int> : true_type {};
  template <> struct is_integral<long> : true_type {};
  template <> struct is_integral<unsigned long> : true_type {};
  template <> struct is_integral<long long> : true_type {};
  template <> struct is_integral<unsigned long long> : true_type {};
  template <class T> constexpr bool is_integral_v = is_integral<T>::value;
#endif

  template <class T> struct is_floating_point
      : bool_constant<
            // Note: standard floating-point types
            is_same_v<float, typename remove_cv<T>::type>
            || is_same_v<double, typename remove_cv<T>::type>
            || is_same_v<long double, typename remove_cv<T>::type>> {};
  template <class T> constexpr bool is_floating_point_v = is_floating_point<T>::value;
  template <class T> struct is_arithmetic
      : bool_constant<is_integral<T>::value || is_floating_point<T>::value> {};
  template <class T> constexpr bool is_arithmetic_v = is_integral_v<T> || is_floating_point_v<T>;
  namespace detail {
    template <typename T, bool = is_arithmetic_v<T>> struct is_signed_impl
        : bool_constant<T(-1) < T(0)> {};
    template <typename T> struct is_signed_impl<T, false> : false_type {};
    template <typename T, bool = is_arithmetic_v<T>> struct is_unsigned_impl
        : bool_constant<T(0) < T(-1)> {};
    template <typename T> struct is_unsigned_impl<T, false> : false_type {};
  }  // namespace detail
  template <typename T> struct is_signed : detail::is_signed_impl<T> {};
  template <typename T> struct is_unsigned : detail::is_unsigned_impl<T> {};
  template <typename T> constexpr bool is_signed_v = is_signed<T>::value;
  template <typename T> constexpr bool is_unsigned_v = is_unsigned<T>::value;
  /// integral type aliases
  using uint = unsigned int;
  // signed
  using i8 = signed char;
  using i16 = signed short;
  using i32 = signed int;
  using i64 = signed long long int;
  static_assert(sizeof(i8) == 1 && sizeof(i16) == 2 && sizeof(i32) == 4 && sizeof(i64) == 8,
                "these signed integers are not of the sizes expected!");
  // unsigned
  using u8 = unsigned char;
  using u16 = unsigned short;
  using u32 = unsigned int;
  using u64 = unsigned long long int;
  static_assert(sizeof(u8) == 1 && sizeof(u16) == 2 && sizeof(u32) == 4 && sizeof(u64) == 8,
                "these unsigned integers are not of the sizes expected!");
  // floating points
  using f32 = float;
  using f64 = double;
  static_assert(sizeof(f32) == 4 && sizeof(f64) == 8,
                "these floating points are not of the sizes expected!");

  // make_unsigned
  template <class T, typename = void> struct make_unsigned;
  template <> struct make_unsigned<bool>;
#define ZS_SPECIALIZE_MAKE_UNSIGNED(FROM, TO) \
  template <> struct make_unsigned<FROM> {    \
    using type = TO;                          \
  };
  ZS_SPECIALIZE_MAKE_UNSIGNED(unsigned char, unsigned char)
  ZS_SPECIALIZE_MAKE_UNSIGNED(unsigned short, unsigned short)
  ZS_SPECIALIZE_MAKE_UNSIGNED(unsigned int, unsigned int)
  ZS_SPECIALIZE_MAKE_UNSIGNED(unsigned long, unsigned long)
  ZS_SPECIALIZE_MAKE_UNSIGNED(unsigned long long, unsigned long long)
  ZS_SPECIALIZE_MAKE_UNSIGNED(signed char, unsigned char)
  ZS_SPECIALIZE_MAKE_UNSIGNED(signed short, unsigned short)
  ZS_SPECIALIZE_MAKE_UNSIGNED(signed int, unsigned int)
  ZS_SPECIALIZE_MAKE_UNSIGNED(signed long, unsigned long)
  ZS_SPECIALIZE_MAKE_UNSIGNED(signed long long, unsigned long long)
  // enum
  template <class T> struct make_unsigned<T, enable_if_type<is_enum_v<T>, void>> {
    using type = typename make_unsigned<underlying_type_t<T>>::type;
  };
  // for the remaining integrals
  template <class T>
  struct make_unsigned<T, enable_if_type<is_integral_v<T> && is_same_v<T, remove_cv_t<T>>, void>> {
    using type = conditional_t<
        sizeof(T) == 1, u8,
        conditional_t<sizeof(T) == 2, u16, conditional_t<sizeof(T) == 4, u32, u64>>>;
    static_assert(sizeof(T) == sizeof(type), "the unsigned type of T does not have the same size");
  };
  // cv-qualified
  template <class T>
  struct make_unsigned<T, enable_if_type<!is_same_v<T, remove_cv_t<T>>,
                                         void_t<typename make_unsigned<remove_cv_t<T>>::type>>> {
    static constexpr bool isConst = is_const_v<T>;
    static constexpr bool isVolatile = is_volatile_v<T>;
    using underlying_t = typename make_unsigned<remove_cv_t<T>>::type;
    using type = conditional_t<
        isConst && isVolatile, add_cv_t<underlying_t>,
        conditional_t<isConst, add_const_t<underlying_t>,
                      conditional_t<isVolatile, add_volatile_t<underlying_t>, underlying_t>>>;
  };
  template <class T> using make_unsigned_t = typename make_unsigned<T>::type;
  static_assert(is_same_v<make_unsigned_t<const volatile int>, volatile const unsigned int>, "???");
  static_assert(is_same_v<make_unsigned_t<volatile long>, volatile unsigned long>, "???");

  // make_signed
  template <class T, typename = void> struct make_signed;
  template <> struct make_signed<bool>;
#define ZS_SPECIALIZE_MAKE_SIGNED(FROM, TO) \
  template <> struct make_signed<FROM> {    \
    using type = TO;                        \
  };
  ZS_SPECIALIZE_MAKE_SIGNED(unsigned char, signed char)
  ZS_SPECIALIZE_MAKE_SIGNED(unsigned short, signed short)
  ZS_SPECIALIZE_MAKE_SIGNED(unsigned int, signed int)
  ZS_SPECIALIZE_MAKE_SIGNED(unsigned long, signed long)
  ZS_SPECIALIZE_MAKE_SIGNED(unsigned long long, signed long long)
  ZS_SPECIALIZE_MAKE_SIGNED(signed char, signed char)
  ZS_SPECIALIZE_MAKE_SIGNED(signed short, signed short)
  ZS_SPECIALIZE_MAKE_SIGNED(signed int, signed int)
  ZS_SPECIALIZE_MAKE_SIGNED(signed long, signed long)
  ZS_SPECIALIZE_MAKE_SIGNED(signed long long, signed long long)
  // enum
  template <class T> struct make_signed<T, enable_if_type<is_enum_v<T>, void>> {
    using type = typename make_signed<underlying_type_t<T>>::type;
  };
  // for the remaining integrals
  template <class T>
  struct make_signed<T, enable_if_type<is_integral_v<T> && is_same_v<T, remove_cv_t<T>>, void>> {
    using type = conditional_t<
        sizeof(T) == 1, i8,
        conditional_t<sizeof(T) == 2, i16, conditional_t<sizeof(T) == 4, i32, i64>>>;
    static_assert(sizeof(T) == sizeof(type), "the signed type of T does not have the same size");
  };
  // cv-qualified
  template <class T>
  struct make_signed<T, enable_if_type<!is_same_v<T, remove_cv_t<T>>,
                                       void_t<typename make_signed<remove_cv_t<T>>::type>>> {
    static constexpr bool isConst = is_const_v<T>;
    static constexpr bool isVolatile = is_volatile_v<T>;
    using underlying_t = typename make_signed<remove_cv_t<T>>::type;
    using type = conditional_t<
        isConst && isVolatile, add_cv_t<underlying_t>,
        conditional_t<isConst, add_const_t<underlying_t>,
                      conditional_t<isVolatile, add_volatile_t<underlying_t>, underlying_t>>>;
  };
  template <class T> using make_signed_t = typename make_signed<T>::type;

  static_assert(is_same_v<make_signed_t<const volatile int>, volatile const int>, "???");
  static_assert(is_same_v<make_signed_t<volatile unsigned long long>, volatile long long>, "???");
  static_assert(is_same_v<make_signed_t<const unsigned long>, const long>, "???");
  // scalar
  template <class T> struct is_pointer : false_type {};
  template <class T> struct is_pointer<T *> : true_type {};
  template <class T> struct is_pointer<T *const> : true_type {};
  template <class T> struct is_pointer<T *volatile> : true_type {};
  template <class T> struct is_pointer<T *const volatile> : true_type {};
  template <class T> constexpr bool is_pointer_v = is_pointer<T>::value;

  template <class T> struct is_member_function_pointer_helper : false_type {};
  template <class T, class U> struct is_member_function_pointer_helper<T U::*> : is_function<T> {};
  template <class T> struct is_member_function_pointer
      : is_member_function_pointer_helper<typename remove_cv<T>::type> {};
  template <class T> constexpr bool is_member_function_pointer_v
      = is_member_function_pointer<T>::value;
  template <class T> struct is_member_pointer_helper : false_type {};
  template <class T, class U> struct is_member_pointer_helper<T U::*> : true_type {};
  template <class T> struct is_member_pointer
      : is_member_pointer_helper<typename remove_cv<T>::type> {};
  template <class T> constexpr bool is_member_pointer_v = is_member_pointer<T>::value;
  template <class T> struct is_member_object_pointer
      : bool_constant<is_member_pointer<T>::value && !is_member_function_pointer<T>::value> {};
  template <class T> constexpr bool is_member_object_pointer_v = is_member_object_pointer<T>::value;

  template <class T> struct is_null_pointer : is_same<nullptr_t, typename remove_cv<T>::type> {};
  template <class T> constexpr bool is_null_pointer_v = is_null_pointer<T>::value;
  template <class T> struct is_scalar
      : bool_constant<is_arithmetic<T>::value || is_enum<T>::value || is_pointer<T>::value
                      || is_member_pointer<T>::value || is_null_pointer<T>::value> {};
  template <class T> constexpr bool is_scalar_v = is_scalar<T>::value;
  // structure
  template <class _Tp> struct is_union : bool_constant<__is_union(_Tp)> {};
  template <class T> constexpr bool is_union_v = is_union<T>::value;
  namespace detail {
    // is_class
    template <class T> bool_constant<!is_union<T>::value> test_is_class(int T::*);
    template <class> false_type test_is_class(...);

  }  // namespace detail
  template <class T> struct is_class : decltype(detail::test_is_class<T>(nullptr)) {};
  // template <class _Tp> struct is_class : bool_constant<__is_class(_Tp)> {};
  template <class T> constexpr bool is_class_v = is_class<T>::value;

  // destructible
  namespace detail {
    // is_destructible
    template <class T, typename = decltype(declval<T &>().~T())>
    true_type test_is_destructible(int);
    template <class> false_type test_is_destructible(...);
    template <class T> constexpr bool is_destructible() noexcept {
      constexpr bool pred0 = is_void_v<T> || is_function_v<T> || is_array_unknown_bounds_v<T>;
      constexpr bool pred1 = is_reference_v<T> || is_scalar_v<T>;
      // gcc11 std impl
      if constexpr (pred0 || pred1 == false) {
        return decltype(test_is_destructible<remove_all_extents_t<T>>(0))::value;
      } else if constexpr (pred0 && !pred1) {
        return false;
      } else if constexpr (!pred0 && pred1) {
        return true;
      } else {
        static_assert(always_false<T>, "T should never be both");
      }
    }
    // is_nothrow_destructible
    template <class T>
    bool_constant<noexcept(declval<T &>().~T())> test_is_nothrow_destructible(int);
    template <class> false_type test_is_nothrow_destructible(...);
    template <class T> constexpr bool is_nothrow_destructible() noexcept {
      constexpr bool pred0 = is_void_v<T> || is_function_v<T> || is_array_unknown_bounds_v<T>;
      constexpr bool pred1 = is_reference_v<T> || is_scalar_v<T>;
      if constexpr (pred0 || pred1 == false) {
        return decltype(test_is_nothrow_destructible<remove_all_extents_t<T>>(0))::value;
      } else if constexpr (pred0 && !pred1) {
        return false;
      } else if constexpr (!pred0 && pred1) {
        return true;
      } else {
        static_assert(always_false<T>, "T should never be both");
      }
    }
  }  // namespace detail
  template <class T> struct is_destructible : bool_constant<detail::is_destructible<T>()> {};
  template <class T> constexpr bool is_destructible_v = is_destructible<T>::value;
  template <class T> struct is_nothrow_destructible
      : bool_constant<detail::is_nothrow_destructible<T>()> {};
  template <class T> constexpr bool is_nothrow_destructible_v = is_nothrow_destructible<T>::value;

  template <class T> struct is_trivially_destructible
      : bool_constant<detail::is_destructible<T>() && __has_trivial_destructor(T)> {};
  template <class T> constexpr bool is_trivially_destructible_v
      = is_trivially_destructible<T>::value;

  // constructible
  template <typename T, typename... Args> struct is_constructible
      : bool_constant<__is_constructible(T, Args...)> {};
  template <typename T, typename... Args> constexpr bool is_constructible_v
      = __is_constructible(T, Args...);
  template <typename T, typename... Args> struct is_nothrow_constructible
      : bool_constant<__is_nothrow_constructible(T, Args...)> {};
  template <typename T, typename... Args> constexpr bool is_nothrow_constructible_v
      = __is_nothrow_constructible(T, Args...);

  template <typename T> struct is_default_constructible : is_constructible<T> {};
  template <typename T> constexpr bool is_default_constructible_v
      = is_default_constructible<T>::value;
  template <typename T> struct is_nothrow_default_constructible
      : bool_constant<__is_nothrow_constructible(T)> {};
  template <typename T> constexpr bool is_nothrow_default_constructible_v
      = __is_nothrow_constructible(T);

  template <typename T, typename = void> struct is_copy_constructible : false_type {};
  template <typename T> struct is_copy_constructible<T, void_t<const T &>>
      : is_constructible<T, const T &> {};
  template <typename T> constexpr bool is_copy_constructible_v = is_copy_constructible<T>::value;
  template <typename T, typename = void> struct is_nothrow_copy_constructible : false_type {};
  template <typename T> struct is_nothrow_copy_constructible<T, void_t<const T &>>
      : is_nothrow_constructible<T, const T &> {};
  template <typename T> constexpr bool is_nothrow_copy_constructible_v
      = is_nothrow_copy_constructible<T>::value;

  template <typename T, typename = void> struct is_move_constructible : false_type {};
  template <typename T> struct is_move_constructible<T, void_t<T &&>> : is_constructible<T, T &&> {
  };
  template <typename T> constexpr bool is_move_constructible_v = is_move_constructible<T>::value;
  template <typename T, typename = void> struct is_nothrow_move_constructible : false_type {};
  template <typename T> struct is_nothrow_move_constructible<T, void_t<T &&>>
      : is_nothrow_constructible<T, T &&> {};
  template <typename T> constexpr bool is_nothrow_move_constructible_v
      = is_nothrow_move_constructible<T>::value;

  // is base of
  namespace details {
    template <typename B> true_type test_ptr_conv(const volatile B *);
    template <typename> false_type test_ptr_conv(const volatile void *);

    template <typename B, typename D> auto test_is_base_of(int)
        -> decltype(test_ptr_conv<B>(static_cast<D *>(nullptr)));
    template <typename, typename> auto test_is_base_of(...)
        -> true_type;  // private or ambiguous base
  }                    // namespace details
  template <typename Base, typename Derived> struct is_base_of
      : bool_constant<is_class_v<Base>
                      && is_class_v<Derived> &&decltype(details::test_is_base_of<Base, Derived>(
                          0))::value> {};
  template <typename Base, typename Derived> constexpr bool is_base_of_v
      = is_base_of<Base, Derived>::value;
  // is_fundamental
  template <class T> struct is_fundamental
      : bool_constant<is_arithmetic<T>::value || is_void<T>::value
                      || is_same_v<nullptr_t, remove_cv_t<T>>> {};
  template <class T> constexpr bool is_fundamental_v
      = is_arithmetic_v<T> || is_void_v<T> || is_same_v<nullptr_t, remove_cv_t<T>>;
  // is_object
  template <class T> struct is_object : bool_constant<is_scalar<T>::value || is_array<T>::value
                                                      || is_union<T>::value || is_class<T>::value> {
  };
  template <class T> constexpr bool is_object_v = is_object<T>::value;

  // is_assignable
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
  template <typename TT, typename T> struct is_nothrow_assignable
      : bool_constant<is_assignable_v<TT, T> && noexcept(declval<TT>() = declval<T>())> {};
  template <typename TT, typename T> constexpr bool is_nothrow_assignable_v
      = is_nothrow_assignable<TT, T>::value;

  template <typename T, typename = void> struct is_copy_assignable : false_type {};
  template <typename T> struct is_copy_assignable<T, void_t<const T &>>
      : is_assignable<T, const T &> {};
  template <typename T> constexpr bool is_copy_assignable_v = is_copy_assignable<T>::value;
  template <typename T, typename = void> struct is_nothrow_copy_assignable : false_type {};
  template <typename T> struct is_nothrow_copy_assignable<T, void_t<const T &>>
      : is_nothrow_assignable<T, const T &> {};
  template <typename T> constexpr bool is_nothrow_copy_assignable_v = is_nothrow_copy_assignable<T>::value;

  template <typename T, typename = void> struct is_move_assignable : false_type {};
  template <typename T> struct is_move_assignable<T, void_t<T &&>> : is_assignable<T, T &&> {};
  template <typename T> constexpr bool is_move_assignable_v = is_move_assignable<T>::value;
  template <typename T, typename = void> struct is_nothrow_move_assignable : false_type {};
  template <typename T> struct is_nothrow_move_assignable<T, void_t<T &&>>
      : is_nothrow_assignable<T, T &&> {};
  template <typename T> constexpr bool is_nothrow_move_assignable_v
      = is_nothrow_move_assignable<T>::value;

  // is_convertible (from cppref)
  namespace detail {
    template <class T> auto test_returnable(int)
        -> decltype(void(static_cast<T (*)()>(nullptr)), true_type{});
    template <class> false_type test_returnable(...);

    template <class From, class To> auto test_implicitly_convertible(int)
        -> decltype(void(declval<void (&)(To)>()(declval<From>())), true_type{});
    template <class, class> false_type test_implicitly_convertible(...);

  }  // namespace detail
  template <class From, class To> struct is_convertible
      : bool_constant<(decltype(detail::test_returnable<To>(0))::value
                       && decltype(detail::test_implicitly_convertible<From, To>(0))::value)
                      || (is_void<From>::value && is_void<To>::value)> {};
  template <class From, class To> constexpr bool is_convertible_v = is_convertible<From, To>::value;

  // common_type (from cppref)
  template <typename... Ts> struct common_type {};
  // 1
  template <class T> struct common_type<T> : common_type<T, T> {};
  namespace detail {
    template <class T1, class T2> using conditional_result_t
        = decltype(false ? declval<T1>() : declval<T2>());

    template <class, class, class = void> struct decay_conditional_result {};
    template <class T1, class T2>
    struct decay_conditional_result<T1, T2, void_t<conditional_result_t<T1, T2>>>
        : decay<conditional_result_t<T1, T2>> {};

    template <class T1, class T2, class = void> struct common_type_2_impl
        : decay_conditional_result<const T1 &, const T2 &> {};
    template <class T1, class T2>
    struct common_type_2_impl<T1, T2, void_t<conditional_result_t<T1, T2>>>
        : decay_conditional_result<T1, T2> {};
  }  // namespace detail
  //////// two types
  template <class T1, class T2> struct common_type<T1, T2>
      : conditional_t<is_same_v<T1, decay_t<T1>> && is_same_v<T2, decay_t<T2>>,
                      detail::common_type_2_impl<T1, T2>, common_type<decay_t<T1>, decay_t<T2>>> {};
  // 3+
  namespace detail {
    template <class AlwaysVoid, class T1, class T2, class... R> struct common_type_multi_impl {};
    template <class T1, class T2, class... R>
    struct common_type_multi_impl<void_t<typename common_type<T1, T2>::type>, T1, T2, R...>
        : common_type<typename common_type<T1, T2>::type, R...> {};
  }  // namespace detail
  template <class T1, class T2, class... R> struct common_type<T1, T2, R...>
      : detail::common_type_multi_impl<void, T1, T2, R...> {};

  template <typename... Ts> using common_type_t = typename common_type<Ts...>::type;

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

  ///
  /// advanced query
  ///
  template <class T> constexpr enable_if_type<is_object<T>::value, T *> addressof(T &arg) noexcept {
    return reinterpret_cast<T *>(&const_cast<char &>(reinterpret_cast<const volatile char &>(arg)));
  }
  template <class T>
  constexpr enable_if_type<!is_object<T>::value, T *> addressof(T &arg) noexcept {
    return &arg;
  }

  template <class T> class reference_wrapper;
  namespace detail {
    struct __invoke_memfun_ref {};
    struct __invoke_memfun_deref {};
    struct __invoke_memobj_ref {};
    struct __invoke_memobj_deref {};
    struct __invoke_other {};

    template <typename _Tp, typename _Tag> struct __result_of_success : wrapt<_Tp> {
      using __invoke_type = _Tag;
    };

    /// ref: ubuntu c++11 header [/usr/include/c++/11/type_traits]
    /// ref: https://github.com/microsoft/STL/stl/inc/type_traits
    template <typename _Tp, typename _Up = remove_cvref_t<_Tp>> struct __inv_unwrap {
      using type = _Tp;
    };
    template <typename _Tp, typename _Up> struct __inv_unwrap<_Tp, reference_wrapper<_Up>> {
      using type = _Up &;
    };
    // callable is a member object
    // [func.require] paragraph 1 bullet 3:
    struct __result_of_memobj_ref_impl {
      template <typename _Fp, typename _Tp1>
      static __result_of_success<decltype(declval<_Tp1>().*declval<_Fp>()), __invoke_memobj_ref>
      _S_test(int);
      template <typename, typename> static failure_type _S_test(...);
    };
    template <typename _MemPtr, typename _Arg> struct __result_of_memobj_ref
        : private __result_of_memobj_ref_impl {
      using type = decltype(_S_test<_MemPtr, _Arg>(0));
    };
    // [func.require] paragraph 1 bullet 4:
    struct __result_of_memobj_deref_impl {
      template <typename _Fp, typename _Tp1>
      static __result_of_success<decltype((*declval<_Tp1>()).*declval<_Fp>()),
                                 __invoke_memobj_deref>
      _S_test(int);
      template <typename, typename> static failure_type _S_test(...);
    };
    template <typename _MemPtr, typename _Arg> struct __result_of_memobj_deref
        : private __result_of_memobj_deref_impl {
      using type = decltype(_S_test<_MemPtr, _Arg>(0));
    };

    template <typename _MemPtr, typename _Arg> struct __result_of_memobj;

    template <typename _Res, typename _Class, typename _Arg>
    struct __result_of_memobj<_Res _Class::*, _Arg> {
      using _Argval = remove_cvref_t<_Arg>;
      using _MemPtr = _Res _Class::*;
      using type =
          typename conditional_t<is_same_v<_Argval, _Class> || is_base_of<_Class, _Argval>::value,
                                 __result_of_memobj_ref<_MemPtr, _Arg>,
                                 __result_of_memobj_deref<_MemPtr, _Arg>>::type;
    };

    // callable is a member func
    // [func.require] paragraph 1 bullet 1:
    struct __result_of_memfun_ref_impl {
      template <typename _Fp, typename _Tp1, typename... _Args>
      static __result_of_success<decltype((declval<_Tp1>().*declval<_Fp>())(declval<_Args>()...)),
                                 __invoke_memfun_ref>
      _S_test(int);
      template <typename...> static failure_type _S_test(...);
    };
    template <typename _MemPtr, typename _Arg, typename... _Args> struct __result_of_memfun_ref
        : private __result_of_memfun_ref_impl {
      using type = decltype(_S_test<_MemPtr, _Arg, _Args...>(0));
    };
    // [func.require] paragraph 1 bullet 2:
    struct __result_of_memfun_deref_impl {
      template <typename _Fp, typename _Tp1, typename... _Args>
      static __result_of_success<decltype(((*declval<_Tp1>())
                                           .*declval<_Fp>())(declval<_Args>()...)),
                                 __invoke_memfun_deref>
      _S_test(int);
      template <typename...> static failure_type _S_test(...);
    };
    template <typename _MemPtr, typename _Arg, typename... _Args> struct __result_of_memfun_deref
        : private __result_of_memfun_deref_impl {
      using type = decltype(_S_test<_MemPtr, _Arg, _Args...>(0));
    };

    template <typename _MemPtr, typename _Arg, typename... _Args> struct __result_of_memfun;
    template <typename _Res, typename _Class, typename _Arg, typename... _Args>
    struct __result_of_memfun<_Res _Class::*, _Arg, _Args...> {
      using _Argval = remove_reference_t<_Arg>;
      using _MemPtr = _Res _Class::*;
      using type = typename conditional_t<is_base_of<_Class, _Argval>::value,
                                          __result_of_memfun_ref<_MemPtr, _Arg, _Args...>,
                                          __result_of_memfun_deref<_MemPtr, _Arg, _Args...>>::type;
    };
    template <typename Fn, typename _Arg, typename... _Args>
    __result_of_memfun<Fn, typename __inv_unwrap<_Arg>::type, _Args...>
    __result_of_memfun_delegate() {
      return {};
    }
    // callable is free func, etc.
    template <typename Fn, typename... Args>
    static __result_of_success<decltype(declval<Fn>()(declval<Args>()...)), __invoke_other>
    invoke_test(int);
    template <typename...> static failure_type invoke_test(...);

    // deduce invoke result
    template <bool IsMemberObjectPtr, bool IsMemberFuncPtr, typename Fn, typename... Args>
    constexpr auto deduce_invoke_result() {
      if constexpr (IsMemberObjectPtr && IsMemberFuncPtr)
        return failure_type{};
      else if constexpr (IsMemberObjectPtr) {
        if constexpr (sizeof...(Args) == 1)
          return typename __result_of_memobj<decay_t<Fn>,
                                             typename __inv_unwrap<Args>::type...>::type{};
        else
          return failure_type{};
      } else if constexpr (IsMemberFuncPtr) {
        if constexpr (sizeof...(Args) > 1)
          return typename decltype(__result_of_memfun_delegate<decay_t<Fn>, Args...>())::type{};
        else
          return failure_type{};
      } else
        return decltype(invoke_test<Fn, Args...>(0)){};
    }
  }  // namespace detail
  /// ref: https://en.cppreference.com/w/cpp/utility/functional
  /// invoke_result
  template <typename Functor, typename... Args> struct invoke_result
      : decltype(detail::deduce_invoke_result<
                 is_member_object_pointer_v<remove_reference_t<Functor>>,
                 is_member_function_pointer_v<remove_reference_t<Functor>>, Functor, Args...>()) {
    // _Fn must be a complete class or an unbounded array
    // each argument type must be a complete class or an unbounded array
  };
  template <typename _Fn, typename... _Args> using invoke_result_t =
      typename invoke_result<_Fn, _Args...>::type;

  /// result_of
  template <typename> struct result_of;
  template <typename _Functor, typename... _ArgTypes> struct result_of<_Functor(_ArgTypes...)>
      : invoke_result<_Functor, _ArgTypes...> {};
  template <typename _Fn> using result_of_t = typename result_of<_Fn>::type;

  /// is_invocable
  // The primary template is used for invalid INVOKE expressions.
  template <typename _Result, typename _Ret, bool = is_void<_Ret>::value, typename = void>
  struct __is_invocable_impl : false_type {};
  // Used for valid INVOKE and INVOKE<void> expressions.
  template <typename _Result, typename _Ret>
  struct __is_invocable_impl<_Result, _Ret,
                             /* is_void<_Ret> = */ true, void_t<typename _Result::type>>
      : true_type {};
  // Used for INVOKE<R> expressions to check the implicit conversion to R.
  template <typename _Result, typename _Ret>
  struct __is_invocable_impl<_Result, _Ret,
                             /* is_void<_Ret> = */ false, void_t<typename _Result::type>> {
  private:
    // The type of the INVOKE expression.
    // Unlike declval, this doesn't add_rvalue_reference.
    static typename _Result::type _S_get();

    // The argument passed may be a different type convertible to _Tp
    template <typename _Tp> static void _S_conv(_Tp);

    // This overload is viable if INVOKE(f, args...) can convert to _Tp.
    template <typename _Tp, typename = decltype(_S_conv<_Tp>(_S_get()))>
    static true_type _S_test(int);
    template <typename _Tp> static false_type _S_test(...);

  public:
    using type = decltype(_S_test<_Ret>(1));
  };
  template <typename _Fn, typename... _ArgTypes> struct is_invocable
      : __is_invocable_impl<invoke_result<_Fn, _ArgTypes...>, void>::type {
    // check complete or unbounded for _Fn, _ArgTypes
  };
  template <typename _Fn, typename... _ArgTypes> constexpr bool is_invocable_v
      = is_invocable<_Fn, _ArgTypes...>::value;
  template <typename _Ret, typename _Fn, typename... _ArgTypes> struct is_invocable_r
      : __is_invocable_impl<invoke_result<_Fn, _ArgTypes...>, _Ret>::type {
    // check complete or unbounded for _Fn, _ArgTypes, _Ret
  };
  template <typename _Ret, typename _Fn, typename... _ArgTypes> constexpr bool is_invocable_r_v
      = is_invocable_r<_Ret, _Fn, _ArgTypes...>::value;

  // check no-throw
  template <typename _Fn, typename _Tp, typename... _Args>
  constexpr bool __call_is_nt(detail::__invoke_memfun_ref) {
    using _Up = typename detail::__inv_unwrap<_Tp>::type;
    return noexcept((declval<_Up>().*declval<_Fn>())(declval<_Args>()...));
  }
  template <typename _Fn, typename _Tp, typename... _Args>
  constexpr bool __call_is_nt(detail::__invoke_memfun_deref) {
    return noexcept(((*declval<_Tp>()).*declval<_Fn>())(declval<_Args>()...));
  }
  template <typename _Fn, typename _Tp> constexpr bool __call_is_nt(detail::__invoke_memobj_ref) {
    using _Up = typename detail::__inv_unwrap<_Tp>::type;
    return noexcept(declval<_Up>().*declval<_Fn>());
  }
  template <typename _Fn, typename _Tp> constexpr bool __call_is_nt(detail::__invoke_memobj_deref) {
    return noexcept((*declval<_Tp>()).*declval<_Fn>());
  }
  template <typename _Fn, typename... _Args> constexpr bool __call_is_nt(detail::__invoke_other) {
    return noexcept(declval<_Fn>()(declval<_Args>()...));
  }
  template <typename _Result, typename _Fn, typename... _Args> struct __call_is_nothrow
      : bool_constant<__call_is_nt<_Fn, _Args...>(typename _Result::__invoke_type{})> {};
  template <typename _Fn, typename... _Args> using __call_is_nothrow_
      = __call_is_nothrow<invoke_result<_Fn, _Args...>, _Fn, _Args...>;
  /// no_throw_invocable
  template <typename _Fn, typename... _Args> struct is_nothrow_invocable
      : bool_constant<is_invocable_v<_Fn, _Args...> && __call_is_nothrow_<_Fn, _Args...>::value> {};
  template <typename _Fn, typename... _Args> constexpr bool is_nothrow_invocable_v
      = is_nothrow_invocable<_Fn, _Args...>::value;

  template <class T, class U = T>
  constexpr T exchange(T &obj,
                       U &&new_value) noexcept(  // since C++23
                                                 // is_nothrow_move_constructible<T>::value &&
      is_nothrow_assignable<T &, U>::value) {
    T old_value = zs::move(obj);
    obj = zs::forward<U>(new_value);
    return old_value;
  }

  namespace detail {
    template <class> constexpr bool is_reference_wrapper_v = false;
    template <class U> constexpr bool is_reference_wrapper_v<reference_wrapper<U>> = true;

    template <class C, class Pointed, class T1, class... Args>
    constexpr decltype(auto) invoke_memptr(Pointed C::*f, T1 &&t1, Args &&...args) {
      if constexpr (is_function<Pointed>::value) {
        if constexpr (is_base_of<C, decay_t<T1>>::value)
          return (forward<T1>(t1).*f)(forward<Args>(args)...);
        else if constexpr (is_reference_wrapper_v<decay_t<T1>>)
          return (t1.get().*f)(forward<Args>(args)...);
        else
          return ((*forward<T1>(t1)).*f)(forward<Args>(args)...);
      } else {
        static_assert(is_object_v<Pointed> && sizeof...(args) == 0);
        if constexpr (is_base_of<C, decay_t<T1>>::value)
          return forward<T1>(t1).*f;
        else if constexpr (is_reference_wrapper_v<decay_t<T1>>)
          return t1.get().*f;
        else
          return (*forward<T1>(t1)).*f;
      }
    }
  }  // namespace detail

  template <class F, class... Args> constexpr invoke_result_t<F, Args...> invoke(
      F &&f, Args &&...args) noexcept(is_nothrow_invocable_v<F, Args...>) {
    if constexpr (is_member_pointer_v<decay_t<F>>)
      return detail::invoke_memptr(f, forward<Args>(args)...);
    else
      return forward<F>(f)(forward<Args>(args)...);
  }

  // reference_wrapper
  /// https://zh.cppreference.com/w/cpp/utility/tuple/make_tuple
  namespace detail {
    template <class T> constexpr T &pass_ref_only(T &t) noexcept { return t; }
    template <class T> void pass_ref_only(T &&) = delete;
  }  // namespace detail
  template <class T> class reference_wrapper {
  public:
    // types
    using type = T;
    // construct/copy/destroy
    template <class U,
              class = decltype(detail::pass_ref_only<T>(declval<U>()),
                               enable_if_t<!is_same_v<reference_wrapper, remove_cvref_t<U>>>())>
    constexpr reference_wrapper(U &&u) noexcept(
        noexcept(detail::pass_ref_only<T>(zs::forward<U>(u))))
        : _ptr(zs::addressof(detail::pass_ref_only<T>(zs::forward<U>(u)))) {}

    reference_wrapper(const reference_wrapper &) noexcept = default;

    // assignment
    reference_wrapper &operator=(const reference_wrapper &x) noexcept = default;

    // access
    constexpr operator T &() const noexcept { return *_ptr; }
    constexpr T &get() const noexcept { return *_ptr; }

    template <class... ArgTypes>
    constexpr invoke_result_t<T &, ArgTypes...> operator()(ArgTypes &&...args) const
        noexcept(is_nothrow_invocable_v<T &, ArgTypes...>) {
      return invoke(get(), forward<ArgTypes>(args)...);
    }

  private:
    T *_ptr;
  };
  template <class T> reference_wrapper(T &) -> reference_wrapper<T>;

  // ref: https://en.cppreference.com/w/cpp/utility/functional/ref
  template <class T> constexpr reference_wrapper<T> ref(T &t) noexcept {
    return reference_wrapper(t);
  }
  template <class T> constexpr reference_wrapper<T> ref(reference_wrapper<T> t) noexcept {
    return reference_wrapper(t.get());
  }
  template <class T> void ref(const T &&) = delete;

  template <class T> constexpr reference_wrapper<const T> cref(const T &t) noexcept {
    return reference_wrapper(t);
  }
  template <class T> constexpr reference_wrapper<const T> cref(reference_wrapper<T> t) noexcept {
    return reference_wrapper(t.get());
  }
  template <class T> void cref(const T &&) = delete;

  /// special_decay = decay + unref
  template <class T> struct unwrap_refwrapper {
    using type = T;
  };
  template <class T> struct unwrap_refwrapper<reference_wrapper<T>> {
    using type = T &;
  };
  template <class T> using special_decay_t = typename unwrap_refwrapper<decay_t<T>>::type;

  template <class T> struct is_refwrapper : false_type {};
  template <class T> struct is_refwrapper<reference_wrapper<T>> : true_type {};
  template <class T> static constexpr bool is_refwrapper_v = is_refwrapper<decay_t<T>>::value;

}  // namespace zs