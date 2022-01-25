#pragma once

#include <functional>
#include <limits>

#include "Meta.h"
#include "Relationship.h"
#include "zensim/tpls/tl/function_ref.hpp"
#include "zensim/types/Function.h"
#include "zensim/types/Optional.h"

namespace zs {

  // gcem alike, shorter alias for std::numeric_limits
  template <typename T> using limits = std::numeric_limits<T>;

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
    using type = decltype(std::declval<MapperF &>()(std::declval<Oprand>()));
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
  template <typename T = void> using plus = std::plus<T>;
  template <typename T = void> using minus = std::minus<T>;
  template <typename T = void> using logical_or = std::logical_or<T>;
  template <typename T = void> using logical_and = std::logical_and<T>;
  template <typename T = void> using multiplies = std::multiplies<T>;
  template <typename T = void> struct getmax {
    template <typename Auto, typename TT = T, enable_if_t<is_same_v<TT, void>> = 0>
    constexpr Auto operator()(const Auto &lhs, const Auto &rhs) const noexcept {
      return lhs > rhs ? lhs : rhs;
    }
    template <typename TT = T, enable_if_t<!is_same_v<TT, void>> = 0>
    constexpr T operator()(const T &lhs, const T &rhs) const noexcept {
      return lhs > rhs ? lhs : rhs;
    }
  };
  template <typename T> struct getmin {
    template <typename Auto, typename TT = T, enable_if_t<is_same_v<TT, void>> = 0>
    constexpr Auto operator()(const Auto &lhs, const Auto &rhs) const noexcept {
      return lhs < rhs ? lhs : rhs;
    }
    template <typename TT = T, enable_if_t<!is_same_v<TT, void>> = 0>
    constexpr T operator()(const T &lhs, const T &rhs) const noexcept {
      return lhs < rhs ? lhs : rhs;
    }
  };

  /// monoid operation for value sequence declaration
  template <typename BinaryOp> struct monoid_op;
  template <typename T> struct monoid_op<plus<T>> {
    static constexpr T e{0};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      return (std::forward<Args>(args) + ...);
    }
  };
  template <typename T> struct monoid_op<multiplies<T>> {
    static constexpr T e{1};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      return (std::forward<Args>(args) * ...);
    }
  };
  template <typename T> struct monoid_op<logical_or<T>> {
    static constexpr bool e{false};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      return (std::forward<Args>(args) || ...);
    }
  };
  template <typename T> struct monoid_op<logical_and<T>> {
    static constexpr bool e{true};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      return (std::forward<Args>(args) && ...);
    }
  };
  template <typename T> struct monoid_op<getmax<T>> {
    static constexpr T e{limits<T>::lowest()};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      T res{e};
      return ((res = res > args ? res : args), ...);
    }
  };
  template <typename T> struct monoid_op<getmin<T>> {
    static constexpr T e{limits<T>::max()};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      T res{e};
      return ((res = res < args ? res : args), ...);
    }
  };

  /// map operation
  struct count_leq {  ///< count less and equal
    template <typename... Tn> constexpr auto operator()(std::size_t M, Tn... Ns) const noexcept {
      if constexpr (sizeof...(Tn) > 0)
        return ((Ns <= M ? 1 : 0) + ...);
      else
        return 0;
    }
  };

  /// monad : construct (T -> M<T>), mbind
  /// functor: mcompose, map, transform
  namespace view {
    template <template <typename> class Functor, typename T, typename R>
    constexpr auto map(const Functor<T> &functor, tl::function_ref<R(T)> f) -> Functor<R> {
      Functor<R> res{functor};
      for (auto &e : res) e = f(e);
      return res;
    }
  }  // namespace view

  namespace action {}  // namespace action

  template <typename T> struct add_optional { using type = optional<T>; };
  template <typename T> struct add_optional<optional<T>> { using type = optional<T>; };

  template <typename T> using add_optional_t = typename add_optional<T>::type;

}  // namespace zs
