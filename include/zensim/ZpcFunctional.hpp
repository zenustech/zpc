#pragma once

#include "zensim/TypeAlias.hpp"

namespace zs {

  /// WIP: supplement
  template <template <class...> class Function, typename Oprand> struct map {
    using type = Function<Oprand>;
  };
  template <template <class...> class Function, template <class...> class Functor, typename... Args>
  struct map<Function, Functor<Args...>> {
    using type = Functor<Function<Args>...>;
  };
  template <template <class...> class Function, typename Functor> using map_t =
      typename map<Function, Functor>::type;

  template <typename MapperF, typename Oprand, bool recursive = true> struct map_op {
    using type = decltype(declval<MapperF &>()(declval<Oprand>()));
  };
  template <typename MapperF, template <class...> class Functor, typename... Args>
  struct map_op<MapperF, Functor<Args...>, true> {
    using type = Functor<typename map_op<MapperF, Args, false>::type...>;
  };
  template <typename MapperF, typename Functor> using map_op_t =
      typename map_op<MapperF, Functor>::type;

  // applicative functor: pure, apply
  // either, apply, join, bind, mcombine, fold

  /// binary operation
  template <typename T = void> struct plus {
    constexpr T operator()(const T &x, const T &y) const { return x + y; }
  };
  template <> struct plus<void> {
    template <typename A, typename B> constexpr auto operator()(A &&x, B &&y) const
        noexcept(noexcept(FWD(x) + FWD(y))) {
      return FWD(x) + FWD(y);
    }
  };
  template <typename T = void> struct minus {
    constexpr T operator()(const T &x, const T &y) const { return x - y; }
  };
  template <> struct minus<void> {
    template <typename A, typename B> constexpr auto operator()(A &&x, B &&y) const
        noexcept(noexcept(FWD(x) - FWD(y))) {
      return FWD(x) - FWD(y);
    }
  };
  template <typename T = void> struct logical_or {
    constexpr T operator()(const T &x, const T &y) const { return x || y; }
  };
  template <> struct logical_or<void> {
    template <typename A, typename B> constexpr auto operator()(A &&x, B &&y) const
        noexcept(noexcept(FWD(x) || FWD(y))) {
      return FWD(x) || FWD(y);
    }
  };
  template <typename T = void> struct logical_and {
    constexpr T operator()(const T &x, const T &y) const { return x && y; }
  };
  template <> struct logical_and<void> {
    template <typename A, typename B> constexpr auto operator()(A &&x, B &&y) const
        noexcept(noexcept(FWD(x) && FWD(y))) {
      return FWD(x) && FWD(y);
    }
  };
  template <typename T = void> struct multiplies {
    constexpr T operator()(const T &x, const T &y) const { return x * y; }
  };
  template <> struct multiplies<void> {
    template <typename A, typename B> constexpr auto operator()(A &&x, B &&y) const
        noexcept(noexcept(FWD(x) * FWD(y))) {
      return FWD(x) * FWD(y);
    }
  };
  template <typename T = void> struct divides {
    constexpr T operator()(const T &x, const T &y) const { return x / y; }
  };
  template <> struct divides<void> {
    template <typename A, typename B> constexpr auto operator()(A &&x, B &&y) const
        noexcept(noexcept(FWD(x) / FWD(y))) {
      return FWD(x) / FWD(y);
    }
  };
  template <typename T = void> struct getmax {
    constexpr T operator()(const T &lhs, const T &rhs) const noexcept(noexcept(lhs > rhs)) {
      return lhs > rhs ? lhs : rhs;
    }
  };
  template <> struct getmax<void> {
    template <typename A, typename B> constexpr auto operator()(A &&x, B &&y) const
        noexcept(noexcept(x > y)) {
      // static_assert(is_same_v<remove_cvref_t<A>, remove_cvref_t<B>>, "x and y should be of the
      // same type, though decorator might differ");
      if (x > y)
        return FWD(x);
      else
        return FWD(y);
    }
  };
  template <typename T = void> struct getmin {
    constexpr T operator()(const T &lhs, const T &rhs) const noexcept(noexcept(lhs < rhs)) {
      return lhs < rhs ? lhs : rhs;
    }
  };
  template <> struct getmin<void> {
    template <typename A, typename B> constexpr auto operator()(A &&x, B &&y) const
        noexcept(noexcept(x < y)) {
      // static_assert(is_same_v<remove_cvref_t<A>, remove_cvref_t<B>>, "x and y should be of the
      // same type, though decorator might differ");
      if (x < y)
        return FWD(x);
      else
        return FWD(y);
    }
  };

  namespace detail {
    template <typename T> struct extract_template_type_argument;
    template <template <class> class TT, typename T> struct extract_template_type_argument<TT<T>> {
      using type = T;
    };
  }  // namespace detail

  namespace detail {
    template <typename T> struct extract_template_type_argument2 {
      using type = void;
    };
    template <template <class, class> class TT, typename T0, typename T1>
    struct extract_template_type_argument2<TT<T0, T1>> {
      using type0 = T0;
      using type1 = T1;
    };
  }  // namespace detail

  /// @brief monoid
  /// @brief custom monoid, user should guarantee that the operator is commutative and
  /// compatible with the identity
  template <typename BinaryOp,
            typename T = typename detail::extract_template_type_argument<BinaryOp>::type,
            typename = void>
  struct monoid {
    template <typename BOp, typename TT> constexpr monoid(BOp op, TT e) : bop{op}, e{e} {}
    ~monoid() = default;

    constexpr T identity() const noexcept { return e; }

    constexpr T operator()() const noexcept { return identity(); }
    template <typename Arg> constexpr T operator()(Arg arg) const noexcept { return arg; }
    template <typename Arg, typename... Args>
    constexpr T operator()(Arg arg, Args... args) const noexcept {
      if constexpr (is_invocable_v<BinaryOp, Arg, Args...>)
        return bop(arg, args...);
      else
        return bop(arg, operator()(args...));
    }

    BinaryOp bop;
    T e;
  };
  template <typename Bop, typename T>
  monoid(Bop, T) -> monoid<remove_cvref_t<Bop>, remove_cvref_t<T>>;

  /// @brief predefined monoids
  template <typename T> struct monoid<plus<T>, T> {
    static_assert(is_arithmetic_v<T>, "T must be an arithmetic type.");
    static constexpr auto identity() noexcept { return 0; }
    template <typename... Args> constexpr T operator()(Args... args) const noexcept {
      return (args + ...);
    }
  };
  template <typename T> struct monoid<multiplies<T>, T> {
    static_assert(is_arithmetic_v<T>, "T must be an arithmetic type.");
    static constexpr T identity() noexcept { return 1; }
    template <typename... Args> constexpr T operator()(Args... args) const noexcept {
      if constexpr (sizeof...(Args) == 0)
        return identity();
      else
        return (args * ...);
    }
  };
  template <typename T> struct monoid<logical_or<T>, T> {
    static_assert(is_arithmetic_v<T>, "T must be an arithmetic type.");
    static constexpr bool identity() noexcept { return false; }
    template <typename... Args> constexpr bool operator()(Args... args) const noexcept {
      return (args || ...);
    }
  };
  template <typename T> struct monoid<logical_and<T>, T> {
    static_assert(is_arithmetic_v<T>, "T must be an arithmetic type.");
    static constexpr bool identity() noexcept { return true; }
    template <typename... Args> constexpr bool operator()(Args... args) const noexcept {
      return (args && ...);
    }
  };
  namespace detail {
    template <typename T> constexpr T deduce_numeric_epsilon() {
      static_assert(is_arithmetic_v<T> && !is_same_v<T, long double>,
                    "T must be an arithmetic type (long double excluded).");
      if constexpr (is_integral_v<T>)
        return 0;
      else if constexpr (is_same_v<T, float>)
        return ZS_FLT_EPSILON;
      else if constexpr (is_same_v<T, double>)
        return ZS_DBL_EPSILON;
      else
        static_assert(always_false<T>, "not implemented for this type!");
    }
    template <typename T> constexpr T deduce_numeric_max() {
      static_assert(is_arithmetic_v<T> && !is_same_v<T, long double>,
                    "T must be an arithmetic type (long double excluded).");
      if constexpr (is_integral_v<T>) {
        if constexpr (is_signed_v<T>)
          return static_cast<T>(~(static_cast<T>(1) << (sizeof(T) * 8 - 1)));
        else
          return ~static_cast<T>(0);
      } else if constexpr (is_same_v<T, float>)
        return ZS_FLT_MAX;
      else if constexpr (is_same_v<T, double>)
        return ZS_DBL_MAX;
      else
        static_assert(always_false<T>, "not implemented for this type!");
    }
    template <typename T> constexpr T deduce_numeric_min() {
      static_assert(is_arithmetic_v<T> && !is_same_v<T, long double>,
                    "T must be an arithmetic type (long double excluded).");
      if constexpr (is_integral_v<T>) {
        if constexpr (is_signed_v<T>)
          return (static_cast<T>(1) << (sizeof(T) * 8 - 1));
        else
          return static_cast<T>(0);
      } else if constexpr (is_same_v<T, float>)
        return ZS_FLT_MIN;
      else if constexpr (is_same_v<T, double>)
        return ZS_DBL_MIN;
      else
        static_assert(always_false<T>, "not implemented for this type!");
    }
    template <typename T> constexpr T deduce_numeric_lowest() {
      static_assert(is_arithmetic_v<T> && !is_same_v<T, long double>,
                    "T must be an arithmetic type (long double excluded).");
      if constexpr (is_integral_v<T>) {
        if constexpr (is_signed_v<T>)
          return static_cast<T>(1) << (sizeof(T) * 8 - 1);
        else
          return static_cast<T>(0);
      } else if constexpr (is_same_v<T, float>)
        return -ZS_FLT_MAX;
      else if constexpr (is_same_v<T, double>)
        return -ZS_DBL_MAX;
      else
        static_assert(always_false<T>, "not implemented for this type!");
    }
    template <typename T> constexpr T deduce_numeric_infinity() {
      static_assert(is_arithmetic_v<T> && !is_same_v<T, long double>,
                    "T must be an arithmetic type (long double excluded).");
      if constexpr (is_integral_v<T>)
        return deduce_numeric_max<T>();
      else if constexpr (is_same_v<T, float>)
        return __builtin_huge_valf();
      else if constexpr (is_same_v<T, double>)
        return __builtin_huge_val();
      else
        static_assert(always_false<T>, "not implemented for this type!");
    }
  }  // namespace detail
  template <typename T> struct monoid<getmax<T>, T> {
    static_assert(is_arithmetic_v<T>, "T must be an arithmetic type.");
    // -infinity() only for floating point
    static constexpr T identity() noexcept { return detail::deduce_numeric_lowest<T>(); }
    template <typename... Args> constexpr T operator()(Args... args) const noexcept {
      T res{identity()};
      return ((res = res > args ? res : args), ...);
    }
  };
  template <typename T> struct monoid<getmin<T>, T> {
    static_assert(is_arithmetic_v<T>, "T must be an arithmetic type.");
    // infinity() only for floating point
    static constexpr T identity() noexcept { return detail::deduce_numeric_max<T>(); }
    template <typename... Args> constexpr T operator()(Args... args) const noexcept {
      T res{identity()};
      return ((res = res < args ? res : args), ...);
    }
  };

  template <typename BinaryOp> constexpr auto make_monoid(wrapt<BinaryOp>) {
    return monoid<remove_cvref_t<BinaryOp>>{};
  }
  template <typename BinaryOp> constexpr auto make_monoid(BinaryOp) {
    return monoid<remove_cvref_t<BinaryOp>>{};
  }
  template <typename BinaryOp, typename T> constexpr auto make_monoid(BinaryOp &&bop, T &&e) {
    return monoid<remove_cvref_t<BinaryOp>, remove_cvref_t<T>>(FWD(bop), FWD(e));
  }

  /// @brief semiring for graph
  /// @note ref: Linear Algebra is the Right Way to Think About Graphs
  enum struct semiring_e {
    // classic linear algebra, strength of all paths
    real_field = 0,
    plus_times = 0,
    // graph connectivity, bfs
    boolean = 1,
    // shortest path
    tropical = 2,
    min_plus = 2,
    // graph matching, longest path
    max_plus = 3,
    // maximal independent set (graph entry is 1)
    min_times = 4,
    max_times = 5,
    //
    custom
  };

  template <typename MultiplyOp, typename Monoid> struct SemiringImpl : MultiplyOp, Monoid {
    using multiply_op = MultiplyOp;
    using monoid_type = Monoid;
    using value_type = decltype(monoid_type::identity());

    template <typename MultiplyOpT, typename MonoidT>
    constexpr SemiringImpl(MultiplyOpT &&mulop, MonoidT &&monoid) noexcept
        : multiply_op(FWD(mulop)), monoid_type(FWD(monoid)) {}
    ~SemiringImpl() = default;
    // identity
    constexpr value_type identity() noexcept { return monoid_type::identity(); }
    // add
    template <typename... Args> constexpr value_type add(Args... args) const noexcept {
      return monoid_type::operator()(args...);
    }
    // multiply is inherited from base_t (i.e. semiring_impl)
    template <typename T0, typename T1> constexpr value_type multiply(T0 a, T1 b) const noexcept {
      return multiply_op::operator()(a, b);
    }
  };

  /// @note Category can be one of the preset 'wrapv<semiring_e>', or a custom mul operator
  template <typename MultiplyOp, typename Monoid, typename = void> struct semiring
      : SemiringImpl<MultiplyOp, Monoid> {
    using base_t = SemiringImpl<MultiplyOp, Monoid>;
    using multiply_op = typename base_t::multiply_op;
    using monoid_type = typename base_t::monoid_type;
    using value_type = decltype(monoid_type::identity());
    using base_t::add;
    using base_t::identity;
    using base_t::multiply;

    template <typename MultiplyOpT, typename MonoidT>
    constexpr semiring(MultiplyOpT &&mulop, MonoidT &&monoid) noexcept
        : base_t(FWD(mulop), FWD(monoid)) {}
    ~semiring() = default;
  };
  template <typename MultiplyOpT, typename MonoidT>
  semiring(MultiplyOpT &&,
           MonoidT &&) -> semiring<remove_cvref_t<MultiplyOpT>, remove_cvref_t<MonoidT>>;

  ///
  template <typename MultiplyOp, typename BinaryOp>
  constexpr auto make_semiring(MultiplyOp &&op, BinaryOp) {
    return semiring{FWD(op), monoid<BinaryOp>{}};
  }

  /// @brief plus_times/ real_field, +.*
  template <typename DomainT = float>
  constexpr auto make_semiring(wrapv<semiring_e::plus_times>, wrapt<DomainT> = {}) {
    return semiring{multiplies<void>{}, make_monoid(plus<DomainT>{})};
  }
  /// @brief boolean, ||.&&
  template <typename DomainT = bool>
  constexpr auto make_semiring(wrapv<semiring_e::boolean>, wrapt<DomainT> = {}) {
    return semiring{logical_and<void>{}, make_monoid(logical_or<DomainT>{})};
  }

  /// helper struct to avoid lowest/max-like identity value calculation overflow
  template <typename MultiplyOp, template <typename> class ReduceOp, typename Domain>
  struct multiplier_for {
    template <typename T0, typename T1> constexpr Domain operator()(T0 &&a, T1 &&b) const {
      if (a == monoid<ReduceOp<remove_cvref_t<T0>>>::identity()
          || b == monoid<ReduceOp<remove_cvref_t<T1>>>::identity())
        return monoid<ReduceOp<remove_cvref_t<Domain>>>::identity();
      else {
        return MultiplyOp{}(FWD(a), FWD(b));
      }
    }
  };
  /// @brief min_plus/ tropical
  template <typename DomainT>
  constexpr auto make_semiring(wrapv<semiring_e::min_plus>, wrapt<DomainT>) {
    return semiring{multiplier_for<plus<void>, getmin, DomainT>{}, make_monoid(getmin<DomainT>{})};
  }
  /// @brief max_plus
  template <typename DomainT>
  constexpr auto make_semiring(wrapv<semiring_e::max_plus>, wrapt<DomainT>) {
    return semiring{multiplier_for<plus<void>, getmax, DomainT>{}, make_monoid(getmax<DomainT>{})};
  }
  /// @brief min_times
  template <typename DomainT>
  constexpr auto make_semiring(wrapv<semiring_e::min_times>, wrapt<DomainT>) {
    return semiring{multiplier_for<multiplies<void>, getmin, DomainT>{},
                    make_monoid(getmin<DomainT>{})};
  }
  /// @brief max_times
  template <typename DomainT>
  constexpr auto make_semiring(wrapv<semiring_e::max_times>, wrapt<DomainT>) {
    return semiring{multiplier_for<multiplies<void>, getmax, DomainT>{},
                    make_monoid(getmax<DomainT>{})};
  }

  // placeholder
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
        static_assert(partial < (detail::deduce_numeric_max<Tn>() - (c0 - '0')) / 10 + 1,
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
    return index_literals::operator""_th < c... > ();
  }

  /// value_seq
  template <typename TypeSeq, typename Indices> using shuffle_t
      = decltype(INST_(TypeSeq).shuffle(INST_(Indices)));
  template <typename TypeSeq, typename Indices> using shuffle_join_t
      = decltype(INST_(TypeSeq).shuffle_join(INST_(Indices)));
  // value_seq (impl)
  template <auto... Ns> struct value_seq : type_seq<integral<decltype(Ns), Ns>...> {
    using base_t = type_seq<integral<decltype(Ns), Ns>...>;
    using indices = typename base_t::indices;
    static constexpr auto count = base_t::count;
    static constexpr auto get_common_type() noexcept {
      if constexpr (base_t::count == 0)
        return wrapt<size_t>{};
      else
        return wrapt<common_type_t<decltype(Ns)...>>{};
    }
    using value_type = typename decltype(get_common_type())::type;
    using iseq = integer_sequence<value_type, (value_type)Ns...>;
    template <typename T> using to_iseq = integer_sequence<T, (T)Ns...>;

    template <zs::size_t I> static constexpr auto value = base_t::template type<I>::value;

    template <zs::size_t N = sizeof...(Ns), enable_if_t<(N == 1)> = 0>
    constexpr operator typename base_t::template type<0>() const noexcept {
      return {};
    }

    constexpr value_seq() noexcept = default;
    template <typename Ti, auto cnt = count, enable_if_t<(cnt > 0)> = 0>
    constexpr value_seq(integer_sequence<Ti, Ns...>) noexcept {}
    template <auto... Ms, enable_if_all<(Ms == Ns)...> = 0>
    constexpr value_seq(wrapv<Ms>...) noexcept {}
    constexpr value_seq(value_seq &&) noexcept = default;
    constexpr value_seq(const value_seq &) noexcept = default;
    constexpr value_seq &operator=(value_seq &&) noexcept = default;
    constexpr value_seq &operator=(const value_seq &) noexcept = default;
    ///
    /// operations
    ///
    template <auto I = 0> constexpr auto get_value(wrapv<I> = {}) const noexcept {
      return typename base_t::template type<I>{};
    }
    template <typename Ti = value_type> constexpr auto get_iseq(wrapt<Ti> = {}) const noexcept {
      return integer_sequence<Ti, (Ti)Ns...>{};
    }
    template <typename BinaryOp> constexpr auto reduce(BinaryOp) const noexcept {
      return wrapv<monoid<BinaryOp>{}(Ns...)>{};
    }
    template <typename UnaryOp, typename BinaryOp>
    constexpr auto reduce(UnaryOp, BinaryOp) const noexcept {
      return wrapv<monoid<BinaryOp>{}(UnaryOp{}(Ns)...)>{};
    }
    template <typename UnaryOp, typename BinaryOp, size_t... Is>
    constexpr auto map_reduce(UnaryOp, BinaryOp, index_sequence<Is...> = indices{}) noexcept {
      return wrapv<monoid<BinaryOp>{}(UnaryOp{}(Is, Ns)...)>{};
    }
    template <typename BinaryOp, auto... Ms>
    constexpr auto compwise(BinaryOp, value_seq<Ms...>) const noexcept {
      return value_seq<BinaryOp{}(Ns, Ms)...>{};
    }
    /// map (Op(i), index_sequence)
    template <typename MapOp, auto... Js>
    constexpr auto map(MapOp, value_seq<Js...>) const noexcept {
      return value_seq<MapOp{}(Js, Ns...)...>{};
    }
    template <typename MapOp, typename Ti, Ti... Js>
    constexpr auto map(MapOp &&op, integer_sequence<Ti, Js...>) const noexcept {
      return map(FWD(op), value_seq<Js...>{});
    }
    template <typename MapOp, auto N> constexpr auto map(MapOp &&op, wrapv<N> = {}) const noexcept {
      return map(FWD(op), make_index_sequence<N>{});
    }
    /// cat
    template <auto... Is> constexpr auto concat(value_seq<Is...>) const noexcept {
      return value_seq<Ns..., Is...>{};
    }
    template <typename Ti, Ti... Is>
    constexpr auto concat(integer_sequence<Ti, Is...>) const noexcept {
      return value_seq<Ns..., Is...>{};
    }
    /// shuffle
    template <auto... Is> constexpr auto shuffle(value_seq<Is...>) const noexcept {
      return base_t::shuffle(index_sequence<Is...>{});
    }
    /// transform
    template <typename UnaryOp> constexpr auto transform(UnaryOp) const noexcept {
      return value_seq<UnaryOp{}(Ns)...>{};
    }
    /// for_each
    template <typename F> constexpr void for_each(F &&f) const noexcept { (f(Ns), ...); }
    /// scan
    template <auto Cate, typename BinaryOp, typename Indices> struct scan_impl;
    template <auto Cate, typename BinaryOp, size_t... Is>
    struct scan_impl<Cate, BinaryOp, index_sequence<Is...>> {
      template <auto I> static constexpr auto get_sum(wrapv<I>) noexcept {
        if constexpr (Cate == 0)
          return wrapv<monoid<BinaryOp>{}((Is < I ? Ns : monoid<BinaryOp>::identity())...)>{};
        else if constexpr (Cate == 1)
          return wrapv<monoid<BinaryOp>{}((Is <= I ? Ns : monoid<BinaryOp>::identity())...)>{};
        else if constexpr (Cate == 2)
          return wrapv<monoid<BinaryOp>{}((Is > I ? Ns : monoid<BinaryOp>::identity())...)>{};
        else
          return wrapv<monoid<BinaryOp>{}((Is >= I ? Ns : monoid<BinaryOp>::identity())...)>{};
      }
      using type = value_seq<decltype(get_sum(wrapv<Is>{}))::value...>;
    };
    template <auto Cate = 0, typename BinaryOp = plus<value_type>>
    constexpr auto scan(BinaryOp bop = {}) const noexcept {
      return typename scan_impl<Cate, BinaryOp, indices>::type{};
    }
  };
  template <typename Ti, Ti... Ns> value_seq(integer_sequence<Ti, Ns...>) -> value_seq<Ns...>;
  template <auto... Ns> value_seq(wrapv<Ns>...) -> value_seq<Ns...>;
  template <auto... Ns> value_seq(value_seq<Ns>...) -> value_seq<Ns...>;

  template <size_t... Ns> constexpr value_seq<Ns...> dim_c{};

  template <typename T> struct static_value_extent;
  template <auto... Ns> struct static_value_extent<value_seq<Ns...>> : wrapv<(Ns * ...)> {};
  template <auto N> struct static_value_extent<wrapv<N>> : wrapv<N> {};

  /// select (constant integral) value (integral_constant<T, N>) by index
  template <zs::size_t I, typename ValueSeq> using select_value =
      typename ValueSeq::template type<I>;
  template <zs::size_t I, auto... Ns> using select_indexed_value
      = select_value<I, value_seq<Ns...>>;

  template <typename, typename> struct gather;
  template <size_t... Is, typename T, T... Ns>
  struct gather<index_sequence<Is...>, integer_sequence<T, Ns...>> {
    using type = integer_sequence<T, select_indexed_value<Is, Ns...>{}...>;
  };
  template <typename Indices, typename ValueSeq> using gather_t =
      typename gather<Indices, ValueSeq>::type;

  template <typename> struct vseq;
  template <auto... Ns> struct vseq<value_seq<Ns...>> {
    using type = value_seq<Ns...>;
  };
  template <typename Ti, Ti... Ns> struct vseq<integer_sequence<Ti, Ns...>> {
    using type = value_seq<Ns...>;
  };
  template <typename Ti, Ti N> struct vseq<integral<Ti, N>> {
    using type = value_seq<N>;
  };
  template <typename Seq> using vseq_t = typename vseq<Seq>::type;

  /** utilities */
  // extract (type / non-type) template argument list
  template <typename T> struct get_ttal;  // extract type template parameter list
  template <template <class...> class T, typename... Args> struct get_ttal<T<Args...>> {
    using type = type_seq<Args...>;
  };
  template <typename T> using get_ttal_t = typename get_ttal<T>::type;

  template <typename T> struct get_nttal;  // extract type template parameter list
  template <template <auto...> class T, auto... Args> struct get_nttal<T<Args...>> {
    using type = value_seq<Args...>;
  };
  template <typename T> using get_nttal_t = typename get_nttal<T>::type;

  /// assemble functor given template argument list
  /// note: recursively unwrap type_seq iff typelist size is one
  template <template <class...> class T, typename... Args> struct assemble {
    using type = T<Args...>;
  };
  template <template <class...> class T, typename... Args> struct assemble<T, type_seq<Args...>>
      : assemble<T, Args...> {};
  template <template <class...> class T, typename... Args> using assemble_t =
      typename assemble<T, Args...>::type;

  /// same functor with a different template argument list
  template <typename...> struct alternative;
  template <template <class...> class T, typename... Ts, typename... Args>
  struct alternative<T<Ts...>, type_seq<Args...>> {
    using type = T<Args...>;
  };
  template <template <class...> class T, typename... Ts, typename... Args>
  struct alternative<T<Ts...>, Args...> {
    using type = T<Args...>;
  };
  template <typename TTAT, typename... Args> using alternative_t =
      typename alternative<TTAT, Args...>::type;

  /// concatenation
  namespace detail {
    struct concatenation_op {
      template <typename... As, typename... Bs>
      constexpr auto operator()(type_seq<As...>, type_seq<Bs...>) const noexcept {
        return type_seq<As..., Bs...>{};
      }
      template <size_t I, typename... SeqT> static constexpr auto seq_func() {
        using T = select_indexed_type<I, SeqT...>;
        return conditional_t<is_type_seq_v<T>, T, type_seq<T>>{};
      }
      template <typename... SeqT> constexpr auto operator()(type_seq<SeqT...>) const noexcept {
        // auto seq_lambda = [](auto I_) {
        //   using T = select_indexed_type<decltype(I_)::value, SeqT...>;
        //   return conditional_t<is_type_seq_v<T>, T, type_seq<T>>{};
        // };
        constexpr size_t N = sizeof...(SeqT);

        if constexpr (N == 0)
          return type_seq<>{};
        else if constexpr (N == 1)
          return seq_func<0, SeqT...>();
        else if constexpr (N == 2)
          return (*this)(seq_func<0, SeqT...>(), seq_func<1, SeqT...>());
        else {
          constexpr size_t halfN = N / 2;
          return (*this)((*this)(type_seq<SeqT...>{}.shuffle(typename build_seq<halfN>::ascend{})),
                         (*this)(type_seq<SeqT...>{}.shuffle(
                             typename build_seq<N - halfN>::template arithmetic<halfN>{})));
        }
      }
    };
  }  // namespace detail
  template <typename... TSeqs> using concatenation_t
      = decltype(declval<detail::concatenation_op>()(type_seq<TSeqs...>{}));

  template <typename... Ts> template <typename... Args>
  constexpr auto type_seq<Ts...>::pair(type_seq<Args...>) const noexcept {
    return type_seq<concatenation_t<Ts, Args>...>{};
  }

  /// permutation / combination
  namespace detail {
    struct compose_op {
      // merger
      template <typename... As, typename... Bs, size_t... Is>
      constexpr auto get_seq(type_seq<As...>, type_seq<Bs...>,
                             index_sequence<Is...>) const noexcept {
        constexpr auto Nb = sizeof...(Bs);
        return type_seq<concatenation_t<select_indexed_type<Is / Nb, As...>,
                                        select_indexed_type<Is % Nb, Bs...>>...>{};
      }
      template <typename... As, typename... Bs>
      constexpr auto operator()(type_seq<As...>, type_seq<Bs...>) const noexcept {
        constexpr auto N = (sizeof...(As)) * (sizeof...(Bs));
        return conditional_t<N == 0, type_seq<>,
                             decltype(get_seq(type_seq<As...>{}, type_seq<Bs...>{},
                                              make_index_sequence<N>{}))>{};
      }
      template <size_t I, typename... SeqT> static constexpr auto seq_func() {
        using T = select_indexed_type<I, SeqT...>;
        return conditional_t<is_type_seq_v<T>, T, type_seq<T>>{};
      }
      /// more general case
      template <typename... SeqT> constexpr auto operator()(type_seq<SeqT...>) const noexcept {
        // constexpr auto seq_lambda = [](auto I_) noexcept {
        //   using T = select_indexed_type<decltype(I_)::value, SeqT...>;
        //   return conditional_t<is_type_seq_v<T>, T, type_seq<T>>{};
        // };
        constexpr size_t N = sizeof...(SeqT);
        if constexpr (N == 0)
          return type_seq<>{};
        else if constexpr (N == 1)
          return map_t<type_seq, decltype(seq_func<0, SeqT...>())>{};
        else if constexpr (N == 2)
          return (*this)(seq_func<0, SeqT...>(), seq_func<1, SeqT...>());
        else if constexpr (N > 2) {
          constexpr size_t halfN = N / 2;
          return (*this)((*this)(type_seq<SeqT...>{}.shuffle(typename build_seq<halfN>::ascend{})),
                         (*this)(type_seq<SeqT...>{}.shuffle(
                             typename build_seq<N - halfN>::template arithmetic<halfN>{})));
        }
      }
    };
  }  // namespace detail
  template <typename... TSeqs> using compose_t
      = decltype(detail::compose_op{}(type_seq<TSeqs...>{}));

  /// join

  /// variadic type template parameters
  template <typename T, template <typename...> class Ref> struct is_type_specialized : false_type {
  };
  template <template <typename...> class Ref, typename... Ts>
  struct is_type_specialized<Ref<Ts...>, Ref> : true_type {};

  /// variadic non-type template parameters
  template <typename T, template <auto...> class Ref> struct is_value_specialized : false_type {};
  template <template <auto...> class Ref, auto... Args>
  struct is_value_specialized<Ref<Args...>, Ref> : true_type {};

  /** direct operations on sequences */
  template <typename> struct seq_tail {
    using type = index_sequence<>;
  };
  template <zs::size_t I, size_t... Is> struct seq_tail<index_sequence<I, Is...>> {
    using type = index_sequence<Is...>;
  };
  template <typename Seq> using seq_tail_t = typename seq_tail<Seq>::type;

}  // namespace zs
