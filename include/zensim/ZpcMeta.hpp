#pragma once
/// can be used in py interop

namespace zs {

  ///
  /// SFINAE
  ///
  template <bool B> struct enable_if;
  template <> struct enable_if<true> { using type = int; };
  template <bool B> using enable_if_t = typename enable_if<B>::type;
  template <bool... Bs> using enable_if_all = typename enable_if<(Bs && ...)>::type;
  template <bool... Bs> using enable_if_any = typename enable_if<(Bs || ...)>::type;

  // conditional
  template <bool B> struct conditional_impl { template <class T, class F> using type = T; };
  template <> struct conditional_impl<false> { template <class T, class F> using type = F; };
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

  template <typename Tn, Tn N> using integral_t = integral_constant<Tn, N>;
  template <typename> struct is_integral : false_type {};
  template <typename T, T v> struct is_integral<integral_t<T, v>> : true_type {};

  constexpr true_type true_c{};
  constexpr false_type false_c{};

  /// @note fundamental type-value wrapper
  template <typename T> struct wrapt { using type = T; };
  // wrap at most 1 layer
  template <typename T> struct wrapt<wrapt<T>> { using type = T; };
  template <typename T> constexpr wrapt<T> wrapt_c{};
  template <auto N> using wrapv = integral_constant<decltype(N), N>;
  template <auto N> constexpr wrapv<N> wrapv_c{};

  template <typename T> struct is_type_wrapper : false_type {};
  template <typename T> struct is_type_wrapper<wrapt<T>> : true_type {};
  template <typename T> constexpr bool is_type_wrapper_v = is_type_wrapper<T>::value;

  template <typename T> struct is_value_wrapper : false_type {};
  template <auto N> struct is_value_wrapper<wrapv<N>> : true_type {};
  template <typename T> constexpr bool is_value_wrapper_v = is_value_wrapper<T>::value;

  template <std::size_t N> using index_t = integral_constant<std::size_t, N>;
  template <std::size_t N> constexpr index_t<N> index_c{};
  ///
  /// type predicates
  ///
  // (C) array
  template <class T> struct is_array : false_type {};
  template <class T> struct is_array<T[]> : true_type {};
  template <class T, std::size_t N> struct is_array<T[N]> : true_type {};
  template <class T> struct is_unbounded_array : false_type {};
  template <class T> struct is_unbounded_array<T[]> : true_type {};
  // const
  template <class T> struct is_const : false_type {};
  template <class T> struct is_const<const T> : true_type {};
  // volatile
  template <class T> struct is_volatile : false_type {};
  template <class T> struct is_volatile<volatile T> : true_type {};
  // reference
  template <class T> struct is_lvalue_reference : false_type {};
  template <class T> struct is_lvalue_reference<T&> : true_type {};
  template <class T> struct is_rvalue_reference : false_type {};
  template <class T> struct is_rvalue_reference<T&&> : true_type {};
  template <class T> struct is_reference : false_type {};
  template <class T> struct is_reference<T&> : true_type {};
  template <class T> struct is_reference<T&&> : true_type {};
  // function
  template <class T> struct is_function
      : integral_constant<bool, !is_const<const T>::value && !is_reference<T>::value> {};
  // is_same
  struct is_same_wrapper_base {
    static constexpr bool is_same(void*) { return false; };
  };
  template <typename T> struct wrapper : is_same_wrapper_base {
    using is_same_wrapper_base::is_same;
    static constexpr bool is_same(wrapper<T>*) { return true; };
  };
  template <typename T1, typename T2> using is_same
      = integral_constant<bool, wrapper<T1>::is_same((wrapper<T2>*)nullptr)>;
  template <typename T1, typename T2> constexpr auto is_same_v = is_same<T1, T2>::value;

  ///
  /// type decoration
  ///
  // remove_reference
  template <class T> struct remove_reference { using type = T; };
  template <class T> struct remove_reference<T&> { using type = T; };
  template <class T> struct remove_reference<T&&> { using type = T; };
  // remove cv
  template <class T> struct remove_cv { using type = T; };
  template <class T> struct remove_cv<const T> { using type = T; };
  template <class T> struct remove_cv<volatile T> { using type = T; };
  template <class T> struct remove_cv<const volatile T> { using type = T; };
  template <class T> struct remove_const { using type = T; };
  template <class T> struct remove_const<const T> { using type = T; };
  template <class T> struct remove_volatile { using type = T; };
  template <class T> struct remove_volatile<volatile T> { using type = T; };
  template <class T> struct remove_cvref {
    using type = typename remove_cv<typename remove_reference<T>::type>::type;
  };
  template <class T> using remove_cvref_t = typename remove_cvref<T>::type;
  template <class T> struct remove_vref {
    using type = typename remove_volatile<typename remove_reference<T>::type>::type;
  };
  template <class T> using remove_vref_t = typename remove_vref<T>::type;
  // add_pointer
  namespace detail {
    template <class T> auto try_add_pointer(int) -> wrapt<typename remove_reference<T>::type*>;
    template <class T> auto try_add_pointer(...) -> wrapt<T>;

  }  // namespace detail
  template <class T> struct add_pointer : decltype(detail::try_add_pointer<T>(0)) {};
  // add reference
  namespace detail {
    template <class T>  // Note that `cv void&` is a substitution failure
    auto try_add_lvalue_reference(int) -> wrapt<T&>;
    template <class T>  // Handle T = cv void case
    auto try_add_lvalue_reference(...) -> wrapt<T>;

    template <class T> auto try_add_rvalue_reference(int) -> wrapt<T&&>;
    template <class T> auto try_add_rvalue_reference(...) -> wrapt<T>;
  }  // namespace detail
  template <class T> struct add_lvalue_reference
      : decltype(detail::try_add_lvalue_reference<T>(0)) {};
  template <class T> struct add_rvalue_reference
      : decltype(detail::try_add_rvalue_reference<T>(0)) {};
  // remove_extent
  template <class T> struct remove_extent { using type = T; };
  template <class T> struct remove_extent<T[]> { using type = T; };
  template <class T, std::size_t N> struct remove_extent<T[N]> { using type = T; };
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
  template <class T> constexpr T&& forward(typename remove_reference<T>::type& t) noexcept {
    return static_cast<T&&>(t);
  }
  template <class T> constexpr T&& forward(typename remove_reference<T>::type&& t) noexcept {
    static_assert(!is_lvalue_reference<T>::value, "Can not forward an rvalue as an lvalue.");
    return static_cast<T&&>(t);
  }

#if 0
  // reference_wrapper
  /// https://zh.cppreference.com/w/cpp/utility/tuple/make_tuple
  namespace detail {
    template <class T> constexpr T& pass_ref_only(T& t) noexcept { return t; }
    template <class T> void pass_ref_only(T&&) = delete;
  }  // namespace detail
  template <class T> class reference_wrapper {
  public:
    // types
    using type = T;
    // construct/copy/destroy
    template <class U,
              class = decltype(detail::pass_ref_only<T>(declval<U>()),
                               enable_if_t<!is_same_v<reference_wrapper, remove_cvref_t<U>>>())>
    constexpr reference_wrapper(U&& u) noexcept(noexcept(detail::pass_ref_only<T>(forward<U>(u))))
        : _ptr(std::addressof(detail::pass_ref_only<T>(forward<U>(u)))) {}

    reference_wrapper(const reference_wrapper&) noexcept = default;

    // assignment
    reference_wrapper& operator=(const reference_wrapper& x) noexcept = default;

    // access
    constexpr operator T&() const noexcept { return *_ptr; }
    constexpr T& get() const noexcept { return *_ptr; }

    template <class... ArgTypes>
    constexpr std::invoke_result_t<T&, ArgTypes...> operator()(ArgTypes&&... args) const
        noexcept(std::is_nothrow_invocable_v<T&, ArgTypes...>) {
      return std::invoke(get(), forward<ArgTypes>(args)...);
    }

  private:
    T* _ptr;
  };
  template <class T> reference_wrapper(T&) -> reference_wrapper<T>;

  /// special_decay = decay + unref
  template <class T> struct unwrap_refwrapper {
    using type = T;
  };
  template <class T> struct unwrap_refwrapper<reference_wrapper<T>> {
    using type = T&;
  };
  template <class T> using special_decay_t =
      typename unwrap_refwrapper<typename decay<T>::type>::type;

  template <class T> struct is_refwrapper {
    static constexpr bool value = false;
  };
  template <class T> struct is_refwrapper<reference_wrapper<T>> {
    static constexpr bool value = true;
  };
  template <class T> static constexpr bool is_refwrapper_v
      = is_refwrapper<typename decay<T>::type>::value;
#endif

}  // namespace zs