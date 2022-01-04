#pragma once

#include <utility>

#include "zensim/Reflection.h"
#include "zensim/math/MathUtils.h"
#include "zensim/meta/ControlFlow.h"
#include "zensim/meta/Meta.h"
#include "zensim/meta/Relationship.h"
#include "zensim/meta/Sequence.h"

namespace zs {

  template <typename T> struct wrapt;
  template <typename... Ts> struct type_seq;
  template <typename... Ts> struct type_seq;
  template <auto... Ns> struct value_seq;
  template <typename> struct tseq;
  template <typename> struct vseq;
  template <typename... Seqs> struct concat;
  template <typename> struct VecInterface;

  template <typename> struct gen_seq_impl;
  template <std::size_t N> using gen_seq = gen_seq_impl<std::make_index_sequence<N>>;

  template <std::size_t I, typename TypeSeq> using select_type = typename TypeSeq::template type<I>;
  template <std::size_t I, typename... Ts> using select_indexed_type
      = select_type<I, type_seq<Ts...>>;

  template <class T> struct unwrap_refwrapper;
  template <class T> using special_decay_t =
      typename unwrap_refwrapper<typename std::decay_t<T>>::type;

/// Jorg Brown, Cppcon2019, reducing template compilation overhead using
/// features from C++11, 14, 17, and 20
#if 0
template <std::size_t I, typename T> struct tuple_value {
  /// by index
  constexpr T &get(std::integral_constant<std::size_t, I>) noexcept {
    return value;
  }
  constexpr T const &
  get(std::integral_constant<std::size_t, I>) const noexcept {
    return value;
  }
  /// by type
  constexpr T &get(wrapt<T>) noexcept { return value; }
  constexpr T const &get(wrapt<T>) const noexcept { return value; }
  T value;
};
#else
  template <std::size_t I, typename T, typename = void> struct tuple_value : T {
    /// by index
    constexpr T &get(std::integral_constant<std::size_t, I>) noexcept { return *this; }
    constexpr T const &get(std::integral_constant<std::size_t, I>) const noexcept { return *this; }
    /// by type
    constexpr T &get(wrapt<T>) noexcept { return *this; }
    constexpr T const &get(wrapt<T>) const noexcept { return *this; }
  };
  template <std::size_t I, typename T> struct tuple_value<
      I, T,
      std::enable_if_t<(
          std::is_fundamental_v<
              T> || std::is_final_v<T> || std::is_same_v<T, void *> || std::is_reference_v<T> || std::is_pointer_v<T>)>> {
    /// by index
    constexpr T &get(std::integral_constant<std::size_t, I>) noexcept { return value; }
    constexpr T const &get(std::integral_constant<std::size_t, I>) const noexcept { return value; }
    /// by type
    constexpr T &get(wrapt<T>) noexcept { return value; }
    constexpr T const &get(wrapt<T>) const noexcept { return value; }
    T value;
  };
#endif

  template <typename, typename> struct tuple_base;
  template <typename... Ts> struct tuple;

  template <class T, class Tuple> constexpr T make_from_tuple(Tuple &&t);
  template <typename... Ts> constexpr auto forward_as_tuple(Ts &&...ts) noexcept;
  template <typename... Ts> constexpr auto tuple_cat(Ts &&...tuples);
  template <typename... Args> constexpr auto make_tuple(Args &&...args);
  template <typename... Ts> constexpr auto tie(Ts &&...ts) noexcept;

  template <std::size_t... Is, typename... Ts>
  struct tuple_base<std::index_sequence<Is...>, type_seq<Ts...>> : tuple_value<Is, Ts>... {
    using tuple_types = tseq<type_seq<Ts...>>;
    static constexpr std::size_t tuple_size = sizeof...(Ts);

    using tuple_value<Is, Ts>::get...;
    template <std::size_t I> constexpr auto &get() noexcept {
      return get(std::integral_constant<std::size_t, I>{});
    }
    template <std::size_t I> constexpr auto const &get() const noexcept {
      return get(std::integral_constant<std::size_t, I>{});
    }
    template <typename T> constexpr auto &get() noexcept { return get(wrapt<T>{}); }
    template <typename T> constexpr auto const &get() const noexcept { return get(wrapt<T>{}); }
    /// custom
    constexpr auto &head() noexcept { return get(std::integral_constant<std::size_t, 0>{}); }
    constexpr auto const &head() const noexcept {
      return get(std::integral_constant<std::size_t, 0>{});
    }
    constexpr auto &tail() noexcept {
      return get(std::integral_constant<std::size_t, tuple_size - 1>{});
    }
    constexpr auto const &tail() const noexcept {
      return get(std::integral_constant<std::size_t, tuple_size - 1>{});
    }
    constexpr decltype(auto) std() const noexcept { return std::forward_as_tuple(get<Is>()...); }
    constexpr decltype(auto) std() noexcept { return std::forward_as_tuple(get<Is>()...); }
    /// iterator
    /// compwise
    template <typename BinaryOp, typename... TTs> constexpr auto compwise(
        BinaryOp &&op,
        const tuple_base<std::index_sequence<Is...>, type_seq<TTs...>> &t) const noexcept {
      return zs::make_tuple(op(get<Is>(), t.template get<Is>())...);
    }
    template <typename BinaryOp, auto... Ns>
    constexpr auto compwise(BinaryOp &&op, vseq<value_seq<Ns...>>) const noexcept {
      return zs::make_tuple(op(get<Is>(), Ns)...);
    }
    /// for_each
    template <typename UnaryOp> constexpr auto for_each(UnaryOp &&op) const noexcept {
      // https://en.cppreference.com/w/cpp/language/eval_order
      // In the evaluation of each of the following four expressions, using
      // the built-in (non-overloaded) operators, there is a sequence point
      // after the evaluation of the expression a. a && b a || b a ? b : c a ,
      // b
      return (op(get<Is>()), ...);
    }
    /// map
    template <typename MapOp, std::size_t... Js>
    constexpr auto map_impl(MapOp &&op, std::index_sequence<Js...>) const noexcept {
      return zs::make_tuple(op(Js, get<Is>()...)...);
    }
    template <std::size_t N, typename MapOp> constexpr auto map(MapOp &&op) const noexcept {
      return map_impl(std::forward<MapOp>(op), gen_seq<N>::ascend());
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
    constexpr auto reduce(BinaryOp &&bop, MonoidOp &&mop, vseq<value_seq<Ns...>>) const noexcept {
      return mop(bop(get<Is>(), Ns)...);
    }
    template <typename BinaryOp, typename MonoidOp, typename... TTs> constexpr auto reduce(
        BinaryOp &&bop, MonoidOp &&mop,
        const tuple_base<std::index_sequence<Is...>, type_seq<TTs...>> &t) const noexcept {
      return mop(bop(get<Is>(), t.template get<Is>())...);
    }
    /// scan
    template <std::size_t I, bool Exclusive, typename BinaryOp, typename Tuple,
              enable_if_t<(I + 1 < tuple_size)> = 0>
    constexpr decltype(auto) prefix_scan_impl(BinaryOp &&op, Tuple &&tup) const noexcept {
      return prefix_scan_impl<I + 1, Exclusive>(
          std::forward<BinaryOp>(op),
          zs::tuple_cat(std::move(tup), zs::make_tuple(op(tup.tail(), get<I - Exclusive>()))));
    }
    template <std::size_t I, bool Exclusive, typename BinaryOp, typename Tuple,
              enable_if_t<(I + 1 >= tuple_size)> = 0>
    constexpr decltype(auto) prefix_scan_impl(BinaryOp &&op, Tuple &&tup) const noexcept {
      return zs::tuple_cat(std::move(tup), zs::make_tuple(op(tup.tail(), get<I - Exclusive>())));
    }
    template <std::size_t I, bool Exclusive, typename BinaryOp, typename Tuple,
              enable_if_t<I != 0> = 0>
    constexpr decltype(auto) suffix_scan_impl(BinaryOp &&op, Tuple &&tup) const noexcept {
      return suffix_scan_impl<I - 1, Exclusive>(
          std::forward<BinaryOp>(op),
          zs::tuple_cat(zs::make_tuple(op(get<I + Exclusive>(), tup.head())), std::move(tup)));
    }
    template <std::size_t I, bool Exclusive, typename BinaryOp, typename Tuple,
              enable_if_t<I == 0> = 0>
    constexpr decltype(auto) suffix_scan_impl(BinaryOp &&op, Tuple &&tup) const noexcept {
      return zs::tuple_cat(zs::make_tuple(op(get<I + Exclusive>(), tup.head())), std::move(tup));
    }

    template <typename BinaryOp, auto tupsize = tuple_size, enable_if_t<(tupsize > 1)> = 0>
    constexpr auto incl_prefix_scan(BinaryOp &&op) const noexcept {
      return prefix_scan_impl<1, false>(std::forward<BinaryOp>(op), zs::make_tuple(this->get<0>()));
    }
    template <typename BinaryOp, auto tupsize = tuple_size, enable_if_t<(tupsize == 1)> = 0>
    constexpr auto incl_prefix_scan(BinaryOp &&op) const noexcept {
      return zs::make_tuple(this->get<0>());
    }

    template <typename BinaryOp, typename T, auto tupsize = tuple_size,
              enable_if_t<(tupsize > 1)> = 0>
    constexpr auto excl_prefix_scan(BinaryOp &&op, T &&e) const noexcept {
      return prefix_scan_impl<1, true>(std::forward<BinaryOp>(op), zs::make_tuple(e));
    }
    template <typename BinaryOp, typename T, auto tupsize = tuple_size,
              enable_if_t<(tupsize == 1)> = 0>
    constexpr auto excl_prefix_scan(BinaryOp &&op, T &&e) const noexcept {
      return zs::make_tuple(e);
    }

    template <typename BinaryOp, auto tupsize = tuple_size, enable_if_t<(tupsize > 1)> = 0>
    constexpr auto incl_suffix_scan(BinaryOp &&op) const noexcept {
      return suffix_scan_impl<tuple_size - 2, false>(std::forward<BinaryOp>(op),
                                                     zs::make_tuple(get<tuple_size - 1>()));
    }
    template <typename BinaryOp, auto tupsize = tuple_size, enable_if_t<(tupsize == 1)> = 0>
    constexpr auto incl_suffix_scan(BinaryOp &&op) const noexcept {
      return zs::make_tuple(get<tuple_size - 1>());
    }

    template <typename BinaryOp, typename T, auto tupsize = tuple_size,
              enable_if_t<(tupsize > 1)> = 0>
    constexpr auto excl_suffix_scan(BinaryOp &&op, T &&e) const noexcept {
      return suffix_scan_impl<tuple_size - 2, true>(std::forward<BinaryOp>(op), zs::make_tuple(e));
    }
    template <typename BinaryOp, typename T, auto tupsize = tuple_size,
              enable_if_t<(tupsize == 1)> = 0>
    constexpr auto excl_suffix_scan(BinaryOp &&op, T &&e) const noexcept {
      return zs::make_tuple(e);
    }
    /// shuffle
    template <typename... Args> constexpr auto shuffle(Args &&...args) const noexcept {
      return zs::make_tuple(get<std::forward<Args>(args)>()...);
    }
    template <auto... Js> constexpr auto shuffle(vseq<value_seq<Js...>>) const noexcept {
      return zs::make_tuple(get<Js>()...);
    }
    template <std::size_t... Js> constexpr auto shuffle(std::index_sequence<Js...>) const noexcept {
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

  template <class... Types> class tuple;

  template <typename... Ts> struct tuple
      : tuple_base<std::index_sequence_for<Ts...>, type_seq<Ts...>> {
    using typename tuple_base<std::index_sequence_for<Ts...>, type_seq<Ts...>>::tuple_types;

    // vec
    template <typename VecT>
    constexpr std::enable_if_t<VecT::extent == sizeof...(Ts), tuple &> operator=(
        const VecInterface<VecT> &v) noexcept {
      assign_impl(v, std::index_sequence_for<Ts...>{});
      return *this;
    }
    // std::array
    template <typename TT, std::size_t dd> constexpr tuple &operator=(const std::array<TT, dd> &v) {
      assign_impl(v, std::index_sequence_for<Ts...>{});
      return *this;
    }
    // c-array
    template <typename Vec>
    constexpr std::enable_if_t<std::is_array_v<Vec>, tuple &> operator=(const Vec &v) {
      assign_impl(v, std::index_sequence_for<Ts...>{});
      return *this;
    }
    // std::tuple
    template <typename... Args> constexpr tuple &operator=(const std::tuple<Args...> &tup) {
      assign_impl(FWD(tup),
                  std::make_index_sequence<zs::math::min(sizeof...(Ts), sizeof...(Args))>{});
      return *this;
    }

  private:
    template <typename VecT, std::size_t... Is>
    constexpr void assign_impl(const VecInterface<VecT> &v, index_seq<Is...>) noexcept {
      ((void)(this->template get<Is>() = v.val(Is)), ...);
    }
    template <typename Vec, std::size_t... Is>
    constexpr auto assign_impl(const Vec &v, index_seq<Is...>) noexcept -> decltype(v[0], void()) {
      ((void)(this->template get<Is>() = v[Is]), ...);
    }
    template <typename... Args, std::size_t... Is>
    constexpr void assign_impl(const std::tuple<Args...> &tup, index_seq<Is...>) noexcept {
      ((void)(this->template get<Is>() = std::get<Is>(tup)), ...);
    }
  };

  template <typename... Args> tuple(Args...) -> tuple<Args...>;

  template <typename> struct is_tuple : std::false_type {};
  template <typename... Ts> struct is_tuple<tuple<Ts...>> : std::true_type {};
  template <typename T> static constexpr bool is_tuple_v = is_tuple<T>::value;

  template <typename> struct is_std_tuple : std::false_type {};
  template <typename... Ts> struct is_std_tuple<std::tuple<Ts...>> : std::true_type {};
  template <typename T> static constexpr bool is_std_tuple_v = is_std_tuple<T>::value;

  /** tuple_size */
  template <typename T> struct tuple_size;
  template <typename... Ts> struct tuple_size<tuple<Ts...>>
      : std::integral_constant<std::size_t, sizeof...(Ts)> {};
  template <typename Tup>
  static constexpr std::enable_if_t<is_tuple_v<Tup>, std::size_t> tuple_size_v
      = tuple_size<Tup>::value;

  /** tuple_element */
  template <std::size_t I, typename T, typename = void> struct tuple_element;
  template <std::size_t I, typename... Ts>
  struct tuple_element<I, tuple<Ts...>, std::enable_if_t<(I < sizeof...(Ts))>> {
    using type = select_type<I, typename tuple<Ts...>::tuple_types>;
  };
  template <std::size_t I, typename Tup> using tuple_element_t
      = std::enable_if_t<is_tuple_v<Tup>, std::enable_if_t<(I < (tuple_size_v<Tup>)),
                                                           typename tuple_element<I, Tup>::type>>;

  /** get */
  template <std::size_t I, typename... Ts>
  constexpr auto const &get(const tuple<Ts...> &t) noexcept {
    return t.template get<I>();
  }
  template <std::size_t I, typename... Ts> constexpr auto &get(tuple<Ts...> &t) noexcept {
    return t.template get<I>();
  }
  template <std::size_t I, typename... Ts> constexpr auto &&get(tuple<Ts...> &&t) noexcept {
    return std::move(t).template get<I>();
  }

  template <typename T, typename... Ts> constexpr T const &get(const tuple<Ts...> &t) noexcept {
    return t.template get<T>();
  }
  template <typename T, typename... Ts> constexpr T &get(tuple<Ts...> &t) noexcept {
    return t.template get<T>();
  }
  template <typename T, typename... Ts> constexpr T &&get(tuple<Ts...> &&t) noexcept {
    return std::move(t).template get<T>();
  }

#if 0
template <typename... Ts>
struct std::tuple_size<tuple<Ts...>>
    : std::integral_constant<std::size_t, sizeof...(Ts)> {};

template <std::size_t I, typename... Ts>
struct std::tuple_element<I, tuple<Ts...>> {
  using type = decltype(std::declval<tuple<Ts...>>().template get<I>());
};
#endif

  /** operations */
  namespace detail {
    template <class F, class Tuple, std::size_t... Is,
              enable_if_t<is_tuple_v<remove_cvref_t<Tuple>>> = 0>
    constexpr decltype(auto) apply_impl(F &&f, Tuple &&t, index_seq<Is...>) {
      // should use constexpr zs::invoke
      FWD(f)(get<Is>(FWD(t))...);
    }
  }  // namespace detail
  template <class F, class Tuple, enable_if_t<is_tuple_v<remove_cvref_t<Tuple>>> = 0>
  constexpr decltype(auto) apply(F &&f, Tuple &&t) {
    return detail::apply_impl(FWD(f), FWD(t),
                              std::make_index_sequence<tuple_size_v<remove_cvref_t<Tuple>>>{});
  }
  template <template <class...> class F, class Tuple,
            enable_if_t<is_tuple_v<remove_cvref_t<Tuple>>> = 0>
  constexpr decltype(auto) apply(assemble_t<F, get_ttal_t<remove_cvref_t<Tuple>>> &&f, Tuple &&t) {
    return detail::apply_impl(FWD(f), FWD(t),
                              std::make_index_sequence<tuple_size_v<remove_cvref_t<Tuple>>>{});
  }

  template <std::size_t... Is, typename... Ts>
  constexpr auto shuffle(index_seq<Is...>, const std::tuple<Ts...> &tup) {
    return std::make_tuple(std::get<Is>(tup)...);
  }
  template <std::size_t... Is, typename... Ts>
  constexpr auto shuffle(index_seq<Is...>, const zs::tuple<Ts...> &tup) {
    return zs::make_tuple(zs::get<Is>(tup)...);
  }

  /** for_each */

  /** make_tuple */
  template <typename... Args> constexpr auto make_tuple(Args &&...args) {
    return zs::tuple<special_decay_t<Args>...>{FWD(args)...};
  }
  /** tie */
  template <typename... Args> constexpr auto tie(Args &...args) {
    return zs::tuple<Args &...>{args...};
  }

  /** forward_as_tuple */
  template <typename... Ts> constexpr auto forward_as_tuple(Ts &&...ts) noexcept {
    return zs::tuple<Ts &&...>{FWD(ts)...};
  }

  /** make_from_tuple */
  namespace tuple_detail_impl {
    template <class T, class Tuple, std::size_t... I>
    constexpr T make_from_tuple_impl(Tuple &&t, std::index_sequence<I...>) {
      return T(zs::get<I>(FWD(t))...);
    }
  }  // namespace tuple_detail_impl

  template <class T, class Tuple, enable_if_t<is_tuple_v<remove_cvref_t<Tuple>>> = 0>
  constexpr T make_from_tuple(Tuple &&t) {
    return tuple_detail_impl::make_from_tuple_impl<T>(
        FWD(t),
        std::make_index_sequence<std::declval<std::remove_reference_t<Tuple>>().tuple_size>{});
  }

  /** tuple_cat */
  namespace tuple_detail_impl {
    template <typename R, auto... Os, auto... Is, typename Tuple>
    constexpr decltype(auto) tuple_cat_impl(vseq<value_seq<Os...>>, vseq<value_seq<Is...>>,
                                            Tuple &&tup) {
      return R{tup.template get<Os>().template get<Is>()...};
    }
  }  // namespace tuple_detail_impl
  template <typename... Ts /*, enable_if_t<(is_tuple<remove_cvref_t<Ts>>::value && ...)> = 0*/>
  constexpr auto tuple_cat(Ts &&...tuples) {
    using Tuple = concat<typename std::remove_reference_t<Ts>::tuple_types...>;
    return tuple_detail_impl::tuple_cat_impl<
        tuple_base<typename Tuple::indices, typename Tuple::type::types>>(
        typename Tuple::outer{}, typename Tuple::inner{}, zs::forward_as_tuple(tuples...));
  }

  // need this because zs::tuple's rvalue deduction not finished
  template <typename T> using capture_t
      = conditional_t<std::is_lvalue_reference<T>{}, std::add_lvalue_reference_t<T>,
                      std::remove_reference_t<T>>;
  template <typename... Ts> constexpr auto fwd_capture(Ts &&...xs) {
    return tuple<capture_t<Ts>...>(FWD(xs)...);
  }
#define FWD_CAPTURE(...) ::zs::fwd_capture(FWD(__VA_ARGS__))

}  // namespace zs
