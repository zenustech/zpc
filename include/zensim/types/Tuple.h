#pragma once

#include <type_traits>
#include <utility>

#include "zensim/Reflection.h"
#include "zensim/math/MathUtils.h"
#include "zensim/meta/Meta.h"
#include "zensim/meta/Sequence.h"

namespace zs {

  template <typename T> struct wrapt;
  template <typename... Ts> struct type_seq;
  template <auto... Ns> struct value_seq;
  template <typename... Seqs> struct concat;
  template <typename> struct VecInterface;

  template <size_t I, typename TypeSeq> using select_type = typename TypeSeq::template type<I>;
  template <size_t I, typename... Ts> using select_indexed_type = select_type<I, type_seq<Ts...>>;

  template <class T> struct unwrap_refwrapper;
  template <class T> using special_decay_t =
      typename unwrap_refwrapper<typename decay<T>::type>::type;

/// Jorg Brown, Cppcon2019, reducing template compilation overhead using
/// features from C++11, 14, 17, and 20
#if 0
template <size_t I, typename T> struct tuple_value {
  /// by index
  constexpr T &get(std::integral_constant<size_t, I>) noexcept {
    return value;
  }
  constexpr T const &
  get(std::integral_constant<size_t, I>) const noexcept {
    return value;
  }
  /// by type
  constexpr T &get(wrapt<T>) noexcept { return value; }
  constexpr T const &get(wrapt<T>) const noexcept { return value; }
  T value;
};
#else
#  if !defined(ZS_COMPILER_MSVC)
  template <size_t I, typename T, typename = void> struct tuple_value : T {
    constexpr tuple_value() = default;
    ~tuple_value() = default;
    template <typename V, enable_if_t<std::is_constructible_v<T, V>> = 0>
    constexpr tuple_value(V &&v) noexcept : T(FWD(v)) {}
    constexpr tuple_value(tuple_value &&) = default;
    constexpr tuple_value(const tuple_value &) = default;
    constexpr tuple_value &operator=(tuple_value &&) = default;
    constexpr tuple_value &operator=(const tuple_value &) = default;

    template <typename V, enable_if_t<std::is_assignable_v<T, V>> = 0>
    constexpr tuple_value &operator=(V &&o) {
      T::operator=(FWD(o));
      return *this;
    }

    /// by index
    constexpr T &get(index_t<I>) &noexcept { return *this; }
    constexpr T &&get(index_t<I>) &&noexcept { return std::move(*this); }
    constexpr const T &get(index_t<I>) const &noexcept { return *this; }
    /// by type
    constexpr T &get(wrapt<T>) &noexcept { return *this; }
    constexpr T &&get(wrapt<T>) &&noexcept { return std::move(*this); }
    constexpr const T &get(wrapt<T>) const &noexcept { return *this; }
  };
#  else
  template <size_t I, typename T, typename = void> struct tuple_value {
    constexpr tuple_value() = default;
    ~tuple_value() = default;
    template <typename V, enable_if_t<std::is_constructible_v<T, V>> = 0>
    constexpr tuple_value(V &&v) noexcept : base(FWD(v)) {}
    constexpr tuple_value(tuple_value &&) = default;
    constexpr tuple_value(const tuple_value &) = default;
    constexpr tuple_value &operator=(tuple_value &&) = default;
    constexpr tuple_value &operator=(const tuple_value &) = default;

    template <typename V, enable_if_t<std::is_assignable_v<T, V>> = 0>
    constexpr tuple_value &operator=(V &&o) {
      base = (FWD(o));
      return *this;
    }

    /// by index
    constexpr T &get(index_t<I>) &noexcept { return base; }
    constexpr T &&get(index_t<I>) &&noexcept { return std::move(base); }
    constexpr const T &get(index_t<I>) const &noexcept { return base; }
    /// by type
    constexpr T &get(wrapt<T>) &noexcept { return base; }
    constexpr T &&get(wrapt<T>) &&noexcept { return std::move(base); }
    constexpr const T &get(wrapt<T>) const &noexcept { return base; }

    T base;
  };
#  endif
  template <size_t I, typename T> struct tuple_value<
      I, T,
      enable_if_type<(std::is_fundamental_v<T> || std::is_final_v<T> || std::is_same_v<T, void *>
                      || std::is_reference_v<T> || std::is_pointer_v<T>)>> {
    constexpr tuple_value() = default;
    ~tuple_value() = default;
    template <typename V, enable_if_t<std::is_constructible_v<T, V>> = 0>
    constexpr tuple_value(V &&v) noexcept : value(FWD(v)) {}
    constexpr tuple_value(tuple_value &&) = default;
    constexpr tuple_value(const tuple_value &) = default;
    constexpr tuple_value &operator=(tuple_value &&) = default;
    constexpr tuple_value &operator=(const tuple_value &) = default;

    template <typename V, enable_if_t<std::is_assignable_v<T, V>> = 0>
    constexpr tuple_value &operator=(V &&o) {
      value = FWD(o);
      return *this;
    }

    /// by index
    constexpr conditional_t<std::is_rvalue_reference_v<T>, T, T &> get(index_t<I>) &noexcept {
      if constexpr (std::is_rvalue_reference_v<T>)
        return std::move(value);
      else
        return value;
    }
    constexpr T &&get(index_t<I>) &&noexcept { return std::move(value); }
    template <bool NonRValRef = !std::is_rvalue_reference_v<T>, enable_if_t<NonRValRef> = 0>
    constexpr decltype(auto) get(index_t<I>) const &noexcept {
      return value;
    }
    /// by type
    constexpr conditional_t<std::is_rvalue_reference_v<T>, T, T &> get(wrapt<T>) &noexcept {
      if constexpr (std::is_rvalue_reference_v<T>)
        return std::move(value);
      else
        return value;
    }
    constexpr T &&get(wrapt<T>) &&noexcept { return std::move(value); }
    template <bool NonRValRef = !std::is_rvalue_reference_v<T>, enable_if_t<NonRValRef> = 0>
    constexpr decltype(auto) get(wrapt<T>) const &noexcept {
      return value;
    }
    T value;
  };
#endif

  template <typename, typename> struct tuple_base;
  template <typename... Ts> struct tuple;

  template <typename> struct is_tuple : false_type {};
  template <typename... Ts> struct is_tuple<tuple<Ts...>> : true_type {};
  template <typename T> static constexpr bool is_tuple_v = is_tuple<T>::value;

  template <typename... Args> constexpr auto make_tuple(Args &&...args);
  template <typename T> struct tuple_size;
  template <typename... Ts> struct tuple_size<tuple<Ts...>>
      : std::integral_constant<size_t, sizeof...(Ts)> {};
  template <typename Tup> constexpr enable_if_type<is_tuple_v<Tup>, size_t> tuple_size_v
      = tuple_size<Tup>::value;

  template <size_t... Is, typename... Ts> struct tuple_base<index_sequence<Is...>, type_seq<Ts...>>
      : tuple_value<Is, Ts>... {
    using tuple_types = type_seq<Ts...>;
    static constexpr size_t tuple_size = sizeof...(Ts);

    constexpr tuple_base() = default;
    ~tuple_base() = default;
    template <typename... Vs, enable_if_t<sizeof...(Vs) == tuple_size> = 0>
    constexpr tuple_base(Vs &&...vs) noexcept : tuple_value<Is, Ts>(FWD(vs))... {}
    constexpr tuple_base(tuple_base &&) = default;
    constexpr tuple_base(const tuple_base &) = default;
    constexpr tuple_base &operator=(tuple_base &&) = default;
    constexpr tuple_base &operator=(const tuple_base &) = default;

    template <typename... Vs, enable_if_all<(std::is_constructible_v<Ts, Vs> && ...)> = 0>
    constexpr tuple_base(const tuple_base<index_sequence<Is...>, type_seq<Vs...>> &o)
        : tuple_value<Is, Ts>(o.get(index_t<Is>{}))... {}
    template <typename... Vs, enable_if_all<(std::is_constructible_v<Ts, Vs> && ...)> = 0>
    constexpr tuple_base(tuple_base<index_sequence<Is...>, type_seq<Vs...>> &&o)
        : tuple_value<Is, Ts>(o.get(index_t<Is>{}))... {}
    template <typename... Vs, enable_if_all<(is_assignable_v<Ts, Vs> && ...)> = 0>
    constexpr tuple_base &operator=(const tuple_base<index_sequence<Is...>, type_seq<Vs...>> &
                                        o) noexcept((std::is_nothrow_assignable_v<Ts, Vs> && ...)) {
      ((get(index_t<Is>{}) = o.get(index_t<Is>{})), ...);
      return *this;
    }
    template <typename... Vs, enable_if_all<(is_assignable_v<Ts, Vs> && ...)> = 0>
    constexpr tuple_base &operator=(tuple_base<index_sequence<Is...>, type_seq<Vs...>> &&
                                        o) noexcept((std::is_nothrow_assignable_v<Ts, Vs> && ...)) {
      ((get(index_t<Is>{}) = o.get(index_t<Is>{})), ...);
      return *this;
    }

    using tuple_value<Is, Ts>::get...;
    template <size_t I> constexpr decltype(auto) get() noexcept { return get(index_t<I>{}); }
    template <size_t I> constexpr decltype(auto) get() const noexcept { return get(index_t<I>{}); }
    template <typename T> constexpr decltype(auto) get() noexcept { return get(wrapt<T>{}); }
    template <typename T> constexpr decltype(auto) get() const noexcept { return get(wrapt<T>{}); }
    /// compare
    constexpr bool operator==(const tuple_base &o) const noexcept {
      return ((get<Is>() == o.get<Is>()) && ...);
    }
    constexpr bool operator!=(const tuple_base &o) const noexcept { return !(operator==(o)); }
    /// custom
    constexpr auto &head() noexcept { return get(index_t<0>{}); }
    constexpr auto const &head() const noexcept { return get(index_t<0>{}); }
    constexpr auto &tail() noexcept { return get(index_t<tuple_size - 1>{}); }
    constexpr auto const &tail() const noexcept { return get(index_t<tuple_size - 1>{}); }
    constexpr decltype(auto) std() const noexcept { return std::forward_as_tuple(get<Is>()...); }
    constexpr decltype(auto) std() noexcept { return std::forward_as_tuple(get<Is>()...); }
    /// iterator
    /// compwise
    template <typename BinaryOp, typename... TTs>
    constexpr auto compwise(BinaryOp &&op, const tuple<TTs...> &t) const noexcept {
      return zs::make_tuple(op(get<Is>(), t.template get<Is>())...);
    }
    template <typename BinaryOp, auto... Ns>
    constexpr auto compwise(BinaryOp &&op, value_seq<Ns...>) const noexcept {
      return zs::make_tuple(op(get<Is>(), Ns)...);
    }
    /// for_each
    template <typename UnaryOp> constexpr auto for_each(UnaryOp &&op) const noexcept {
      // https://en.cppreference.com/w/cpp/language/eval_order
      // In the evaluation of each of the following four expressions, using
      // the built-in (non-overloaded) operators, there is a sequence point
      // after the evaluation of the expression a. a && b a || b a ? b : c a , b
      return (op(get<Is>()), ...);
    }
    /// map
    template <typename MapOp, size_t... Js>
    constexpr auto map_impl(MapOp &&op, index_sequence<Js...>) const noexcept {
      return zs::make_tuple(op(Js, get<Is>()...)...);
    }
    template <size_t N, typename MapOp> constexpr auto map(MapOp &&op) const noexcept {
      return map_impl(forward<MapOp>(op), gen_seq<N>::ascend());
    }
    /// reduce
    template <typename MonoidOp> constexpr auto reduce(MonoidOp &&op) const noexcept {
      return op(get<Is>()...);
    }
    template <typename UnaryOp, typename MonoidOp>
    constexpr auto reduce(UnaryOp &&uop, MonoidOp &&mop) const noexcept {
      return mop(uop(get<Is>())...);
    }
    template <typename BinaryOp, typename MonoidOp, auto... Ns>
    constexpr auto reduce(BinaryOp &&bop, MonoidOp &&mop, value_seq<Ns...>) const noexcept {
      return mop(bop(get<Is>(), Ns)...);
    }
    template <typename BinaryOp, typename MonoidOp, typename... TTs> constexpr auto reduce(
        BinaryOp &&bop, MonoidOp &&mop,
        const tuple_base<index_sequence<Is...>, type_seq<TTs...>> &t) const noexcept {
      return mop(bop(get<Is>(), t.template get<Is>())...);
    }
    /// shuffle
    template <typename... Args> constexpr auto shuffle(Args &&...args) const noexcept {
      return zs::make_tuple(get<FWD(args)>()...);
    }
    template <auto... Js> constexpr auto shuffle(value_seq<Js...>) const noexcept {
      return zs::make_tuple(get<Js>()...);
    }
    template <size_t... Js> constexpr auto shuffle(index_sequence<Js...>) const noexcept {
      return zs::make_tuple(get<Js>()...);
    }
    /// transform
    template <typename UnaryOp> constexpr auto transform(UnaryOp &&op) const noexcept {
      return zs::make_tuple(op(get<Is>())...);
    }
    ///
    constexpr auto initializer() const noexcept { return std::initializer_list(get<Is>()...); }

    constexpr operator tuple<Ts...>() const noexcept { return *this; }
  };

  template <typename... Ts> struct tuple : tuple_base<index_sequence_for<Ts...>, type_seq<Ts...>> {
    using base_t = tuple_base<index_sequence_for<Ts...>, type_seq<Ts...>>;
    using tuple_types = typename base_t::tuple_types;
    template <size_t I> using tuple_element_t = select_indexed_type<I, Ts...>;

    constexpr tuple() = default;
    ~tuple() = default;
    template <typename... Vs, enable_if_t<sizeof...(Vs) == sizeof...(Ts)> = 0>
    constexpr tuple(Vs &&...vs) noexcept : base_t(FWD(vs)...) {}
    constexpr tuple(tuple &&) = default;
    constexpr tuple(const tuple &) = default;
    constexpr tuple &operator=(tuple &&) = default;
    constexpr tuple &operator=(const tuple &) = default;

    template <typename... Vs, enable_if_t<(std::is_constructible_v<Ts, Vs> && ...)> = 0>
    constexpr tuple(const tuple<Vs...> &o) : base_t(o) {}
    template <typename... Vs, enable_if_t<(std::is_constructible_v<Ts, Vs> && ...)> = 0>
    constexpr tuple(tuple<Vs...> &&o) : base_t(move(o)) {}
    template <typename... Vs, enable_if_t<is_assignable_v<type_seq<Ts...>, type_seq<Vs...>>> = 0>
    constexpr tuple &operator=(const tuple<Vs...> &o) {
      base_t::operator=(o);
      return *this;
    }
    template <typename... Vs, enable_if_t<is_assignable_v<type_seq<Ts...>, type_seq<Vs...>>> = 0>
    constexpr tuple &operator=(tuple<Vs...> &&o) {
      base_t::operator=(move(o));
      return *this;
    }

    // vec
    template <typename VecT>
    constexpr enable_if_type<VecT::extent == sizeof...(Ts), tuple &> operator=(
        const VecInterface<VecT> &v) noexcept {
      assign_impl(v, index_sequence_for<Ts...>{});
      return *this;
    }
    // std::array
    template <typename TT, size_t dd> constexpr tuple &operator=(const std::array<TT, dd> &v) {
      assign_impl(v, index_sequence_for<Ts...>{});
      return *this;
    }
    // c-array
    template <typename Vec>
    constexpr enable_if_type<std::is_array_v<Vec>, tuple &> operator=(const Vec &v) {
      assign_impl(v, index_sequence_for<Ts...>{});
      return *this;
    }
    // std::tuple
    template <typename... Args> constexpr tuple &operator=(const std::tuple<Args...> &tup) {
      assign_impl(FWD(tup), make_index_sequence<zs::math::min(sizeof...(Ts), sizeof...(Args))>{});
      return *this;
    }

  private:
    template <typename VecT, size_t... Is>
    constexpr void assign_impl(const VecInterface<VecT> &v, index_sequence<Is...>) noexcept {
      ((void)(this->template get<Is>() = v.val(Is)), ...);
    }
    template <typename Vec, size_t... Is>
    constexpr auto assign_impl(const Vec &v, index_sequence<Is...>) noexcept
        -> decltype(v[0], void()) {
      ((void)(this->template get<Is>() = v[Is]), ...);
    }
    template <typename... Args, size_t... Is>
    constexpr void assign_impl(const std::tuple<Args...> &tup, index_sequence<Is...>) noexcept {
      ((void)(this->template get<Is>() = std::get<Is>(tup)), ...);
    }
  };

  template <typename... Args> tuple(Args...) -> tuple<Args...>;

  template <typename> struct is_std_tuple : false_type {};
  template <typename... Ts> struct is_std_tuple<std::tuple<Ts...>> : true_type {};
  template <typename T> static constexpr bool is_std_tuple_v = is_std_tuple<T>::value;

  /** tuple_element */
  template <size_t I, typename T, typename = void> struct tuple_element;
  template <size_t I, typename... Ts>
  struct tuple_element<I, tuple<Ts...>, enable_if_type<(I < sizeof...(Ts))>> {
    using type = select_type<I, typename tuple<Ts...>::tuple_types>;
  };
  template <size_t I, typename Tup> using tuple_element_t
      = enable_if_type<is_tuple_v<Tup>, enable_if_type<(I < (tuple_size_v<Tup>)),
                                                       typename tuple_element<I, Tup>::type>>;

  /** operations */

  /** get */
  template <size_t I, typename... Ts> constexpr decltype(auto) get(const tuple<Ts...> &t) noexcept {
    return t.template get<I>();
  }
  template <size_t I, typename... Ts> constexpr decltype(auto) get(tuple<Ts...> &t) noexcept {
    return t.template get<I>();
  }
  template <size_t I, typename... Ts> constexpr decltype(auto) get(tuple<Ts...> &&t) noexcept {
    return move(t).template get<I>();
  }

  template <typename T, typename... Ts>
  constexpr decltype(auto) get(const tuple<Ts...> &t) noexcept {
    return t.template get<T>();
  }
  template <typename T, typename... Ts> constexpr decltype(auto) get(tuple<Ts...> &t) noexcept {
    return t.template get<T>();
  }
  template <typename T, typename... Ts> constexpr decltype(auto) get(tuple<Ts...> &&t) noexcept {
    return move(t).template get<T>();
  }

  /** make_tuple */
  template <typename... Args> constexpr auto make_tuple(Args &&...args) {
    return zs::tuple<special_decay_t<Args>...>{FWD(args)...};
  }
  template <typename T, size_t... Is>
  constexpr auto make_uniform_tuple(T &&v, index_sequence<Is...>) noexcept {
    return make_tuple((Is ? v : v)...);
  }
  template <size_t N, typename T> constexpr auto make_uniform_tuple(T &&v) noexcept {
    return make_uniform_tuple(FWD(v), make_index_sequence<N>{});
  }

  /** linear_to_multi */
  template <auto... Ns, size_t... Is, enable_if_all<(Ns > 0)...> = 0>
  constexpr auto index_to_coord(size_t I, value_seq<Ns...> vs, index_sequence<Is...>) {
    constexpr auto N = sizeof...(Ns);
    using Tn = typename value_seq<Ns...>::value_type;
    // using RetT = typename gen_seq<N>::template uniform_types_t<tuple, Tn>;
    constexpr auto exsuf = vs.template scan<2>(multiplies<size_t>{});
    Tn bases[N]{exsuf.get_value(wrapv<Is>()).value...};
    Tn cs[N]{};
    for (size_t i = 0; i != N; ++i) {
      cs[i] = I / bases[i];
      I -= bases[i] * cs[i];
    }
    return zs::make_tuple(cs[Is]...);
  }
  template <auto... Ns, enable_if_all<(Ns > 0)...> = 0>
  constexpr auto index_to_coord(size_t I, value_seq<Ns...> vs) {
    return index_to_coord(I, vs, make_index_sequence<sizeof...(Ns)>{});
  }

  /** forward_as_tuple */
  template <typename... Ts> constexpr auto forward_as_tuple(Ts &&...ts) noexcept {
    return zs::tuple<Ts &&...>{FWD(ts)...};
  }

  /** tuple_cat */
  namespace tuple_detail_impl {
    struct count_leq {  ///< count less and equal
      template <typename... Tn> constexpr auto operator()(size_t M, Tn... Ns) const noexcept {
        if constexpr (sizeof...(Tn) > 0)
          return ((Ns <= M ? 1 : 0) + ...);
        else
          return 0;
      }
    };
    /// concat
    template <typename... Tuples> struct concat {
      static_assert((is_tuple_v<remove_cvref_t<Tuples>> && ...),
                    "concat should only take zs::tuple type template params!");
      using counts = value_seq<remove_cvref_t<Tuples>::tuple_types::count...>;
      static constexpr auto length = counts{}.reduce(plus<size_t>{}).value;
      using indices = typename gen_seq<length>::ascend;
      using outer
          = decltype(counts{}.template scan<1, plus<size_t>>().map(count_leq{}, wrapv<length>{}));
      using inner = decltype(vseq_t<indices>{}.compwise(
          minus<size_t>{}, counts{}.template scan<0, plus<size_t>>().shuffle(outer{})));
      // using types = decltype(type_seq<typename
      // remove_cvref_t<Tuples>::tuple_types...>{}.shuffle(outer{}).shuffle_join(inner{}));
      template <auto... Os, auto... Is>
      static constexpr auto get_ret_type(value_seq<Os...>, value_seq<Is...>) {
        // precisely extract types from these tuples
        return type_seq<typename select_indexed_type<
            Os, remove_reference_t<Tuples>...>::template tuple_element_t<Is>...>{};
      }
      // https://en.cppreference.com/w/cpp/utility/tuple/tuple_cat
      using types = decltype(get_ret_type(outer{}, inner{}));
    };
    template <typename R, auto... Os, auto... Is, typename Tuple>
    constexpr decltype(auto) tuple_cat_impl(value_seq<Os...>, value_seq<Is...>, Tuple &&tup) {
      return R{get<Is>(get<Os>(tup))...};
    }
  }  // namespace tuple_detail_impl

  constexpr auto tuple_cat() noexcept { return tuple<>{}; }

  template <auto... Is, typename... Ts> constexpr auto tuple_cat(value_seq<Is...>, Ts &&...tuples) {
    auto tup = zs::forward_as_tuple(FWD(tuples)...);
    return tuple_cat(get<Is>(tup)...);
  }
  template <typename... Ts> constexpr auto tuple_cat(Ts &&...tuples) {
    if constexpr ((!zs::is_tuple_v<remove_cvref_t<Ts>> || ...)) {
      constexpr auto trans = [](auto &&param) -> decltype(auto) {
        if constexpr (zs::is_tuple_v<RM_CVREF_T(param)>)
          return FWD(param);
        else if constexpr (is_refwrapper_v<decltype(param)>) {  // reference
          return zs::tuple<decltype(param.get())>(param.get());
        } else
          return zs::make_tuple(FWD(param));
      };
      return tuple_cat(trans(FWD(tuples))...);
    } else {
      constexpr auto marks = value_seq<(remove_cvref_t<Ts>::tuple_size > 0 ? 1 : 0)...>{};
      if constexpr (marks.reduce(logical_and<bool>{})) {
        // using helper = concat<typename remove_reference_t<Ts>::tuple_types...>;
        using helper = tuple_detail_impl::concat<Ts...>;
        return tuple_detail_impl::tuple_cat_impl<assemble_t<tuple, typename helper::types>>(
            typename helper::outer{}, typename helper::inner{},
            zs::forward_as_tuple(FWD(tuples)...));
      } else {
        constexpr auto N = marks.reduce(plus<int>{}).value;
        constexpr auto offsets = marks.scan();  // exclusive scan
        constexpr auto tags = marks.pair(offsets);
        constexpr auto seq
            = tags.filter(typename vseq_t<typename gen_seq<N>::ascend>::template to_iseq<int>{});
        return tuple_cat(seq, FWD(tuples)...);
      }
    }
  }

  template <typename TupA, typename TupB,
            enable_if_t<(is_tuple_v<TupA> || is_tuple_v<remove_cvref_t<TupB>>)> = 0>
  constexpr auto operator+(TupA &&tupa, TupB &&tupb) {
    return tuple_cat(FWD(tupa), FWD(tupb));
  }

  /** apply */
  namespace detail {
    template <class F, class Tuple, size_t... Is,
              enable_if_t<is_tuple_v<remove_cvref_t<Tuple>>> = 0>
    constexpr decltype(auto) apply_impl(F &&f, Tuple &&t, index_sequence<Is...>) {
      // should use constexpr zs::invoke
      return f(get<Is>(t)...);
    }
  }  // namespace detail
  template <class F, class Tuple, enable_if_t<is_tuple_v<remove_cvref_t<Tuple>>> = 0>
  constexpr decltype(auto) apply(F &&f, Tuple &&t) {
    return detail::apply_impl(FWD(f), FWD(t),
                              make_index_sequence<tuple_size_v<remove_cvref_t<Tuple>>>{});
  }

  template <template <class...> class F, class Tuple,
            enable_if_t<is_tuple_v<remove_cvref_t<Tuple>>> = 0>
  constexpr decltype(auto) apply(assemble_t<F, get_ttal_t<remove_cvref_t<Tuple>>> &&f, Tuple &&t) {
    return detail::apply_impl(FWD(f), FWD(t),
                              make_index_sequence<tuple_size_v<remove_cvref_t<Tuple>>>{});
  }

  template <size_t... Is, typename... Ts>
  constexpr auto shuffle(index_sequence<Is...>, const std::tuple<Ts...> &tup) {
    return std::make_tuple(std::get<Is>(tup)...);
  }
  template <size_t... Is, typename... Ts>
  constexpr auto shuffle(index_sequence<Is...>, const zs::tuple<Ts...> &tup) {
    return zs::make_tuple(zs::get<Is>(tup)...);
  }

  /** tie */
  template <typename... Args> constexpr auto tie(Args &...args) noexcept {
    return zs::tuple<Args &...>{args...};
  }

  /** make_from_tuple */
  namespace tuple_detail_impl {
    template <class T, class Tuple, size_t... Is>
    constexpr T make_from_tuple_impl(Tuple &&t, index_sequence<Is...>) {
      return T{get<Is>(t)...};
    }
  }  // namespace tuple_detail_impl

  template <class T, class Tuple, enable_if_t<is_tuple_v<remove_cvref_t<Tuple>>> = 0>
  constexpr T make_from_tuple(Tuple &&t) {
    return tuple_detail_impl::make_from_tuple_impl<T>(
        FWD(t), make_index_sequence<tuple_size_v<remove_cvref_t<Tuple>>>{});
  }

  // need this because zs::tuple's rvalue deduction not finished
  template <typename T> using capture_t
      = conditional_t<std::is_lvalue_reference<T>{}, std::add_lvalue_reference_t<T>,
                      remove_reference_t<T>>;
  template <typename... Ts> constexpr auto fwd_capture(Ts &&...xs) {
    return tuple<capture_t<Ts>...>(FWD(xs)...);
  }
#define FWD_CAPTURE(...) ::zs::fwd_capture(FWD(__VA_ARGS__))

}  // namespace zs

namespace std {
  template <typename... Ts> struct tuple_size<zs::tuple<Ts...>>
      : std::integral_constant<size_t, zs::tuple_size_v<zs::tuple<Ts...>>> {};
  template <size_t I, typename... Ts> struct tuple_element<I, zs::tuple<Ts...>> {
    using type = zs::tuple_element_t<I, zs::tuple<Ts...>>;
  };
}  // namespace std
