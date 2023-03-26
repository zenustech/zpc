#pragma once

#include <functional>
#include <limits>

#include "Meta.h"
#include "Relationship.h"
#include "zensim/zpc_tpls/tl/function_ref.hpp"

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

  struct static_plus {
    template <typename T = int> constexpr auto e() const noexcept {
      if constexpr (std::is_arithmetic_v<T>)
        return (T)0;
      else
        return 0;
    }
    template <typename... Args> constexpr auto operator()(Args &&...args) const noexcept {
      if constexpr (sizeof...(Args) == 0)
        return e();
      else  // default
        return (FWD(args) + ...);
    }
  };
  struct static_multiplies {
    template <typename T = int> constexpr auto e() const noexcept {
      if constexpr (std::is_arithmetic_v<T>)
        return (T)1;
      else  // default
        return 1;
    }
    template <typename... Args> constexpr auto operator()(Args &&...args) const noexcept {
      if constexpr (sizeof...(Args) == 0)
        return e();
      else
        return (FWD(args) * ...);
    }
  };
  struct static_minus {
    template <typename TA, typename TB> constexpr auto operator()(TA a, TB b) const noexcept {
      return a - b;
    }
  };
  template <bool SafeMeasure = false> struct static_divides {
    template <typename TA, typename TB> constexpr auto operator()(TA a, TB b) const noexcept {
      if constexpr (std::is_arithmetic_v<TB>) {
        if constexpr (SafeMeasure) {
          constexpr auto eps = (TB)128 * limits<TB>::epsilon();
          return (b >= -eps && b <= eps) ? a : a / b;
        } else
          return a / b;
      } else
        return a / b;
    }
  };

  /// @brief monoid
  /// @note BinaryOp should be commutative
  template <typename BinaryOp, typename = void> struct monoid_op;
  template <typename T> struct monoid_op<plus<T>, std::enable_if_t<std::is_fundamental_v<T>>> {
    static constexpr T e{0};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      return (std::forward<Args>(args) + ...);
    }
  };
  template <typename T>
  struct monoid_op<multiplies<T>, std::enable_if_t<std::is_fundamental_v<T>>> {
    static constexpr T e{1};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      return (std::forward<Args>(args) * ...);
    }
  };
  template <typename T>
  struct monoid_op<logical_or<T>, std::enable_if_t<std::is_fundamental_v<T>>> {
    static constexpr bool e{false};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      return (std::forward<Args>(args) || ...);
    }
  };
  template <typename T>
  struct monoid_op<logical_and<T>, std::enable_if_t<std::is_fundamental_v<T>>> {
    static constexpr bool e{true};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      return (std::forward<Args>(args) && ...);
    }
  };
  template <typename T> struct monoid_op<getmax<T>, std::enable_if_t<std::is_fundamental_v<T>>> {
    // -infinity() only for floating point
    static constexpr T e{limits<T>::lowest()};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      T res{e};
      return ((res = res > args ? res : args), ...);
    }
  };
  template <typename T> struct monoid_op<getmin<T>, std::enable_if_t<std::is_fundamental_v<T>>> {
    // infinity() only for floating point
    static constexpr T e{limits<T>::max()};
    template <typename... Args> constexpr T operator()(Args &&...args) const noexcept {
      T res{e};
      return ((res = res < args ? res : args), ...);
    }
  };

  /// @brief semiring for graph
  /// @note ref: Linear Algebra is the Right Way to Think About Graphs
  enum struct semiring_e {
    // classic linear algebra
    real_field = 0,
    plus_times = 0,
    // graph connectivity
    boolean = 1,
    // shortest path
    tropical = 2,
    min_plus = 2,
    // graph matching
    max_plus = 3,
    // maximal independent set
    min_times = 4
  };
  /// @note Derived class should provide: domain_type/ value_type/ bindary_op/ multiply(domain_type,
  /// domain_type)
  template <typename Derived> struct SemiringInterface {
    /// the identity of \"add\" is not the annihilator of \"multiply\"
    template <typename T = Derived> static constexpr auto identity() {
      using monoid = monoid_op<typename T::binary_op>;
      return monoid::e;
    }
    template <typename T = Derived>
    static constexpr auto add(const typename T::value_type &a, const typename T::value_type &b) {
      using monoid = monoid_op<typename T::binary_op>;
      return monoid{}(a, b);
    }
  };
  template <semiring_e category, typename Domain, typename = void> struct semiring_op;

  /// @brief plus_times/ real_field, +.*
  template <typename DomainT> struct semiring_op<semiring_e::plus_times, DomainT,
                                                 std::enable_if_t<std::is_fundamental_v<DomainT>>>
      : SemiringInterface<semiring_op<semiring_e::plus_times, DomainT,
                                      std::enable_if_t<std::is_fundamental_v<DomainT>>>> {
    using domain_type = DomainT;
    using value_type = domain_type;
    using binary_op = plus<value_type>;
    static constexpr value_type multiply(const domain_type &a, const domain_type &b) {
      return a * b;
    }
  };
  /// @brief boolean, ||.&&
  template <typename DomainT>
  struct semiring_op<semiring_e::boolean, DomainT, std::enable_if_t<std::is_integral_v<DomainT>>>
      : SemiringInterface<semiring_op<semiring_e::boolean, DomainT,
                                      std::enable_if_t<std::is_integral_v<DomainT>>>> {
    using domain_type = DomainT;
    using value_type = bool;
    using binary_op = logical_or<value_type>;
    static constexpr value_type multiply(const domain_type &a, const domain_type &b) {
      return a && b;
    }
  };
  /// @brief min_plus/ tropical
  template <typename DomainT> struct semiring_op<semiring_e::min_plus, DomainT,
                                                 std::enable_if_t<std::is_fundamental_v<DomainT>>>
      : SemiringInterface<semiring_op<semiring_e::min_plus, DomainT,
                                      std::enable_if_t<std::is_fundamental_v<DomainT>>>> {
    using domain_type = DomainT;
    using value_type = domain_type;
    using binary_op = getmin<value_type>;
    static constexpr value_type multiply(const domain_type &a, const domain_type &b) {
      constexpr auto e = monoid_op<getmin<domain_type>>::e;
      if (a == e || b == e)
        return e;
      else
        return a + b;
    }
  };
  /// @brief max_plus
  template <typename DomainT> struct semiring_op<semiring_e::max_plus, DomainT,
                                                 std::enable_if_t<std::is_fundamental_v<DomainT>>>
      : SemiringInterface<semiring_op<semiring_e::max_plus, DomainT,
                                      std::enable_if_t<std::is_fundamental_v<DomainT>>>> {
    using domain_type = DomainT;
    using value_type = domain_type;
    using binary_op = getmax<value_type>;
    static constexpr value_type multiply(const domain_type &a, const domain_type &b) {
      constexpr auto e = monoid_op<getmax<domain_type>>::e;
      if (a == e || b == e)
        return e;
      else
        return a + b;
    }
  };
  /// @brief min_times
  template <typename DomainT> struct semiring_op<semiring_e::min_times, DomainT,
                                                 std::enable_if_t<std::is_fundamental_v<DomainT>>>
      : SemiringInterface<semiring_op<semiring_e::min_times, DomainT,
                                      std::enable_if_t<std::is_fundamental_v<DomainT>>>> {
    using domain_type = DomainT;
    using value_type = domain_type;
    using binary_op = getmin<value_type>;
    static constexpr value_type multiply(const domain_type &a, const domain_type &b) {
      constexpr auto e = monoid_op<getmin<domain_type>>::e;
      if (a == e || b == e)
        return e;
      else
        return a * b;
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
  namespace vw {
    template <template <typename> class Functor, typename T, typename R>
    constexpr auto map(const Functor<T> &functor, tl::function_ref<R(T)> f) -> Functor<R> {
      Functor<R> res{functor};
      for (auto &e : res) e = f(e);
      return res;
    }
  }  // namespace vw

  namespace action {}  // namespace action

}  // namespace zs
