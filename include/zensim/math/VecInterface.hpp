#pragma once
#include <tuple>

#include "MathUtils.h"
#include "zensim/meta/Meta.h"
#include "zensim/meta/Sequence.h"

namespace zs {

  template <typename, typename, typename> struct indexer_impl;

  /// indexer
  template <typename Tn, Tn... Ns, std::size_t... StorageOrders, std::size_t... Is>
  struct indexer_impl<integer_seq<Tn, Ns...>, index_seq<StorageOrders...>, index_seq<Is...>> {
    static_assert(sizeof...(Ns) == sizeof...(StorageOrders), "dimension mismatch");
    static_assert(std::is_integral_v<Tn>, "index type is not an integral");
    static constexpr int dim = sizeof...(Ns);
    using index_type = Tn;
    static constexpr index_type extent = (Ns * ...);
    using extents = integer_seq<Tn, Ns...>;
    using storage_orders = value_seq<select_indexed_value<Is, StorageOrders...>::value...>;
    using lookup_orders = value_seq<storage_orders::template index<wrapv<Is>>::value...>;

    using storage_extents_impl
        = integer_seq<index_type, select_indexed_value<StorageOrders, Ns...>::value...>;
    using storage_bases = value_seq<excl_suffix_mul(Is, storage_extents_impl{})...>;
    using lookup_bases = value_seq<
        storage_bases::template type<lookup_orders::template type<Is>::value>::value...>;

    template <std::size_t I, enable_if_t<(I < dim)> = 0>
    static constexpr index_type storage_range() noexcept {
      return select_indexed_value<storage_orders::template type<I>::value, Ns...>::value;
    }
    template <std::size_t I, enable_if_t<(I < dim)> = 0>
    static constexpr index_type range() noexcept {
      return select_indexed_value<I, Ns...>::value;
    }

    static constexpr index_type offset(std::enable_if_t<Is == Is, index_type>... is) noexcept {
      return ((is * lookup_bases::template type<Is>::value) + ...);
    }
    template <std::size_t... Js, typename... Args>
    static constexpr index_type offset_impl(index_seq<Js...>, Args&&... args) noexcept {
      return (... + ((index_type)args * lookup_bases::template type<Js>::value));
    }
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    static constexpr index_type offset(Args&&... args) noexcept {
      return offset_impl(std::index_sequence_for<Args...>{}, FWD(args)...);
    }
  };

  template <typename Tn, Tn... Ns> using indexer
      = indexer_impl<integer_seq<Tn, Ns...>, std::make_index_sequence<sizeof...(Ns)>,
                     std::make_index_sequence<sizeof...(Ns)>>;

  template <typename Orders, typename Tn, Tn... Ns> using ordered_indexer
      = indexer_impl<integer_seq<Tn, Ns...>, Orders, std::make_index_sequence<sizeof...(Ns)>>;

  template <typename T, typename Dims> struct vec_impl;

  template <typename Derived> struct VecInterface {
#define DECLARE_VEC_INTERFACE_ATTRIBUTES                                                       \
  using value_type = typename Derived::value_type;                                             \
  using index_type = typename Derived::index_type;                                             \
  using extents = typename Derived::extents; /*not necessarily same as indexer_type::extents*/ \
  using dims = typename Derived::dims;                                                         \
  using indexer_type = typename Derived::indexer_type;                                         \
  constexpr index_type extent = indexer_type::extent;                                          \
  constexpr auto dim = indexer_type::dim;

    using vec_type = Derived;

    template <typename VecT = Derived> static constexpr bool is_access_lref
        = std::is_lvalue_reference_v<decltype(std::declval<VecT&>().val(0))>;
    template <typename VecT = Derived> using variant_vec_type =
        typename VecT::template variant_vec<typename VecT::value_type, typename VecT::extents>;
    template <typename VecT = Derived> static constexpr bool is_variant_writable
        = std::is_lvalue_reference_v<decltype(std::declval<variant_vec_type<VecT>&>().val(0))>;

    constexpr auto data() noexcept { return static_cast<Derived*>(this)->do_data(); }
    constexpr auto data() volatile noexcept {
      return static_cast<volatile Derived*>(this)->do_data();
    }
    constexpr auto data() const noexcept { return static_cast<const Derived*>(this)->do_data(); }
    constexpr auto data() const volatile noexcept {
      return static_cast<const volatile Derived*>(this)->do_data();
    }
    /// property query
    template <std::size_t I, typename VecT = Derived, enable_if_t<(I < VecT::dim)> = 0>
    static constexpr typename VecT::index_type range() noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      return select_value<I, vseq<extents>>::value;
    }

    ///
    /// entry access
    ///
    // ()
    template <typename VecT = Derived, typename... Tis,
              enable_if_all<sizeof...(Tis) <= VecT::dim, (std::is_integral_v<Tis> && ...)> = 0>
    constexpr decltype(auto) operator()(Tis... is) noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      return static_cast<Derived*>(this)->operator()((is)...);
    }
    template <typename VecT = Derived, typename... Tis,
              enable_if_all<sizeof...(Tis) <= VecT::dim, (std::is_integral_v<Tis> && ...)> = 0>
    constexpr decltype(auto) operator()(Tis... is) const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      return static_cast<const Derived*>(this)->operator()((is)...);
    }
    // val (one dimension index)
    template <typename Ti, enable_if_t<std::is_integral_v<Ti>> = 0>
    constexpr decltype(auto) val(Ti i) noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      return static_cast<Derived*>(this)->do_val(i);
    }
    template <typename Ti, enable_if_t<std::is_integral_v<Ti>> = 0>
    constexpr decltype(auto) val(Ti i) const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      return static_cast<const Derived*>(this)->do_val(i);
    }
    // [] (one dimension index)
    template <typename Ti, enable_if_t<std::is_integral_v<Ti>> = 0>
    constexpr decltype(auto) operator[](Ti is) noexcept {
      return static_cast<Derived*>(this)->operator[](is);
    }
    template <typename Ti, enable_if_t<std::is_integral_v<Ti>> = 0>
    constexpr decltype(auto) operator[](Ti is) const noexcept {
      return static_cast<const Derived*>(this)->operator[](is);
    }
    // tuple as index
    template <typename VecT = Derived, typename... Ts,
              enable_if_all<sizeof...(Ts) <= VecT::dim,
                            (std::is_integral_v<remove_cvref_t<Ts>>, ...)> = 0>
    constexpr decltype(auto) operator()(const std::tuple<Ts...>& is) noexcept {
      return std::apply(static_cast<Derived&>(*this), is);
    }
    template <typename VecT = Derived, typename... Ts,
              enable_if_all<(sizeof...(Ts) <= VecT::dim),
                            (std::is_integral_v<remove_cvref_t<Ts>>, ...)> = 0>
    constexpr decltype(auto) operator()(const std::tuple<Ts...>& is) const noexcept {
      return std::apply(static_cast<const Derived&>(*this), is);
    }

    ///
    /// construction
    ///

    ///
    /// arithmetic operators
    /// ref: https://github.com/cemyuksel/cyCodeBase/blob/master/cyIVector.h
    ///
    //!@name Unary operators
    constexpr auto operator-() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = -this->val(i);
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_integral_v<typename VecT::value_Type>> = 0>
    constexpr auto operator!() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<bool, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = !static_cast<bool>(this->val(i));
      return r;
    }
    template <typename VecT = Derived, enable_if_t<VecT::dim == 2> = 0>
    constexpr auto transpose() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      constexpr auto M = select_value<0, vseq<extents>>::value;
      constexpr auto N = select_value<1, vseq<extents>>::value;
      using extentsT = std::integer_sequence<index_type, N, M>;
      typename Derived::template variant_vec<value_type, extentsT> r{};
      for (index_type i = 0; i != M; ++i)
        for (index_type j = 0; j != N; ++j) r(j, i) = (*this)(i, j);
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto prod() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      value_type res{1};
      for (index_type i = 0; i != extent; ++i) res *= this->val(i);
      return res;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto sum() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      value_type res{0};
      for (index_type i = 0; i != extent; ++i) res += this->val(i);
      return res;
      // return res.sum();
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto l2NormSqr() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      value_type res{0};
      for (index_type i = 0; i != extent; ++i) res += this->val(i) * this->val(i);
      return res;
      // return res.sum();
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto infNormSqr() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      value_type res{0};
      for (index_type i = 0; i != extent; ++i)
        if (value_type sqr = this->val(i) * this->val(i); sqr > res) res = sqr;
      return res;
    }

    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto length() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      value_type sqrNorm = l2NormSqr();
      using T = conditional_t<std::is_floating_point_v<value_type>, value_type,
                              conditional_t<(sizeof(value_type) >= 8), double, float>>;
      return math::sqrtNewtonRaphson(static_cast<T>(sqrNorm));
      // return gcem::sqrt(sqrNorm);
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto norm() const noexcept {
      return length();
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto normalized() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      const auto len = length();
      typename Derived::template variant_vec<RM_CVREF_T(len), extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = this->val(i) / len;
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto max() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      value_type res{this->val(0)};
      for (index_type i = 1; i != extent; ++i)
        if (this->val(i) > res) res = this->val(i);
      return res;
    }
    template <typename VecT = Derived,
              enable_if_all<std::is_arithmetic_v<typename VecT::value_type>, VecT::dim == 1,
                            VecT::extent == 3> = 0>
    constexpr auto orthogonal() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      using T = conditional_t<std::is_floating_point_v<value_type>, value_type,
                              conditional_t<(sizeof(value_type) >= 8), double, float>>;
      T x = gcem::abs(this->val(0));
      T y = gcem::abs(this->val(1));
      T z = gcem::abs(this->val(2));
      using V = typename Derived::template variant_vec<value_type, extents>;
      auto other = x < y ? (x < z ? V{1, 0, 0} : V{0, 0, 1}) : (y < z ? V{0, 1, 0} : V{0, 0, 1});
      return cross(other);
    }
    //!@name Coefficient-wise math funcs
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto abs() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i)
        r.val(i) = this->val(i) > 0 ? this->val(i) : -this->val(i);
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto log() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = gcem::log(this->val(i));
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto log1p() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = gcem::log1p(this->val(i));
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto square() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = this->val(i) * this->val(i);
      return r;
    }
    template <typename VecT = Derived,
              enable_if_t<std::is_fundamental_v<typename VecT::value_type>> = 0>
    constexpr auto cube() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      typename Derived::template variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i)
        r.val(i) = this->val(i) * this->val(i) * this->val(i);
      return r;
    }

    //!@name Binary operators
    // scalar
#define DEFINE_VEC_OP_SCALAR(OP)                                                  \
  template <typename TT, typename VecT = Derived,                                 \
            enable_if_t<std::is_convertible<typename VecT::value_type, TT>::value \
                        && std::is_fundamental_v<TT>> = 0>                        \
  friend constexpr auto operator OP(VecInterface const& e, TT const v) noexcept { \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                              \
    using R = math::op_result_t<value_type, TT>;                                  \
    typename Derived::template variant_vec<R, extents> r{};                       \
    for (index_type i = 0; i != extent; ++i) r.val(i) = (R)e.val(i) OP((R)v);     \
    return r;                                                                     \
  }                                                                               \
  template <typename TT, typename VecT = Derived,                                 \
            enable_if_t<std::is_convertible<typename VecT::value_type, TT>::value \
                        && std::is_fundamental_v<TT>> = 0>                        \
  friend constexpr auto operator OP(TT const v, VecInterface const& e) noexcept { \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                              \
    using R = math::op_result_t<value_type, TT>;                                  \
    typename Derived::template variant_vec<R, extents> r{};                       \
    for (index_type i = 0; i != extent; ++i) r.val(i) = (R)v OP((R)e.val(i));     \
    return r;                                                                     \
  }
    DEFINE_VEC_OP_SCALAR(+)
    DEFINE_VEC_OP_SCALAR(-)
    DEFINE_VEC_OP_SCALAR(*)
    DEFINE_VEC_OP_SCALAR(/)

// scalar integral
#define DEFINE_VEC_OP_SCALAR_INTEGRAL(OP)                                                       \
  template <                                                                                    \
      typename TT, typename VecT = Derived,                                                     \
      enable_if_all<std::is_integral_v<typename VecT::value_type>, std::is_integral_v<TT>> = 0> \
  friend constexpr auto operator OP(VecInterface const& e, TT const v) noexcept {               \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                            \
    using R = math::op_result_t<value_type, TT>;                                                \
    typename Derived::template variant_vec<R, extents> r{};                                     \
    for (index_type i = 0; i != extent; ++i) r.val(i) = (R)e.val(i) OP((R)v);                   \
    return r;                                                                                   \
  }                                                                                             \
  template <                                                                                    \
      typename TT, typename VecT = Derived,                                                     \
      enable_if_all<std::is_integral_v<typename VecT::value_type>, std::is_integral_v<TT>> = 0> \
  friend constexpr auto operator OP(TT const v, VecInterface const& e) noexcept {               \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                            \
    using R = math::op_result_t<value_type, TT>;                                                \
    typename Derived::template variant_vec<R, extents> r{};                                     \
    for (index_type i = 0; i != extent; ++i) r.val(i) = (R)v OP((R)e.val(i));                   \
    return r;                                                                                   \
  }
    DEFINE_VEC_OP_SCALAR_INTEGRAL(&)
    DEFINE_VEC_OP_SCALAR_INTEGRAL(|)
    DEFINE_VEC_OP_SCALAR_INTEGRAL(^)
    DEFINE_VEC_OP_SCALAR_INTEGRAL(>>)
    DEFINE_VEC_OP_SCALAR_INTEGRAL(<<)

    // vector
#define DEFINE_VEC_OP_VECTOR(OP)                                                             \
  template <typename OtherVecT, typename VecT = Derived,                                     \
            enable_if_t<is_same_v<typename OtherVecT::extents, typename VecT::extents>> = 0> \
  friend constexpr auto operator OP(VecInterface const& lhs,                                 \
                                    VecInterface<OtherVecT> const& rhs) noexcept {           \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                         \
    using R = math::op_result_t<value_type, typename OtherVecT::value_type>;                 \
    typename Derived::template variant_vec<R, extents> r{};                                  \
    for (index_type i = 0; i != extent; ++i) r.val(i) = (R)lhs.val(i) OP((R)rhs.val(i));     \
    return r;                                                                                \
  }
    DEFINE_VEC_OP_VECTOR(+)
    DEFINE_VEC_OP_VECTOR(-)
    DEFINE_VEC_OP_VECTOR(/)

#define DEFINE_VEC_OP_VECTOR_GENERAL(OP)                                                     \
  template <typename OtherVecT, typename VecT = Derived,                                     \
            enable_if_t<is_same_v<typename OtherVecT::extents, typename VecT::extents>> = 0> \
  friend constexpr auto operator OP(VecInterface const& lhs,                                 \
                                    VecInterface<OtherVecT> const& rhs) noexcept {           \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                         \
    using R = math::op_result_t<value_type, typename OtherVecT::value_type>;                 \
    typename Derived::template variant_vec<R, extents> r{};                                  \
    for (index_type i = 0; i != extent; ++i) r.val(i) = (R)lhs.val(i) OP((R)rhs.val(i));     \
    return r;                                                                                \
  }
    DEFINE_VEC_OP_VECTOR_GENERAL(*)

    // vector integral
#define DEFINE_VEC_OP_VECTOR_INTEGRAL(OP)                                                 \
  template <typename OtherVecT, typename VecT = Derived,                                  \
            enable_if_all<is_same_v<typename OtherVecT::extents, typename VecT::extents>, \
                          std::is_integral_v<typename OtherVecT::value_type>,             \
                          std::is_integral_v<typename VecT::value_type>> = 0>             \
  friend constexpr auto operator OP(VecInterface const& lhs,                              \
                                    VecInterface<OtherVecT> const& rhs) noexcept {        \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                      \
    using R = math::op_result_t<value_type, typename OtherVecT::value_type>;              \
    typename Derived::template variant_vec<R, extents> r{};                               \
    for (index_type i = 0; i != extent; ++i) r.val(i) = (R)lhs.val(i) OP((R)rhs.val(i));  \
    return r;                                                                             \
  }
    DEFINE_VEC_OP_VECTOR_INTEGRAL(&)
    DEFINE_VEC_OP_VECTOR_INTEGRAL(|)
    DEFINE_VEC_OP_VECTOR_INTEGRAL(^)
    DEFINE_VEC_OP_VECTOR_INTEGRAL(>>)
    DEFINE_VEC_OP_VECTOR_INTEGRAL(<<)

//!@name Assignment operators
// scalar
#define DEFINE_VEC_OP_SCALAR_ASSIGN(OP)                                                     \
  template <typename TT, typename VecT = Derived, bool IsAssignable = is_access_lref<VecT>, \
            enable_if_all<std::is_convertible_v<typename VecT::value_type, TT>,             \
                          std::is_fundamental_v<TT>, IsAssignable> = 0>                     \
  constexpr Derived& operator OP##=(TT&& v) noexcept {                                      \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                        \
    using R = math::op_result_t<value_type, TT>;                                            \
    for (index_type i = 0; i != extent; ++i) this->val(i) = (R)this->val(i) OP((R)v);       \
    return static_cast<Derived&>(*this);                                                    \
  }
    DEFINE_VEC_OP_SCALAR_ASSIGN(+)
    DEFINE_VEC_OP_SCALAR_ASSIGN(-)
    DEFINE_VEC_OP_SCALAR_ASSIGN(*)
    DEFINE_VEC_OP_SCALAR_ASSIGN(/)

    // scalar integral
#define DEFINE_VEC_OP_SCALAR_INTEGRAL_ASSIGN(OP)                                                 \
  template <typename TT, typename VecT = Derived, bool IsAssignable = is_access_lref<VecT>,      \
            enable_if_all<std::is_integral_v<typename VecT::value_type>, std::is_integral_v<TT>, \
                          IsAssignable> = 0>                                                     \
  constexpr Derived& operator OP##=(TT&& v) noexcept {                                           \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                             \
    using R = math::op_result_t<value_type, TT>;                                                 \
    for (index_type i = 0; i != extent; ++i) this->val(i) = (R)this->val(i) OP((R)v);            \
    return static_cast<Derived&>(*this);                                                         \
  }
    DEFINE_VEC_OP_SCALAR_INTEGRAL_ASSIGN(&)
    DEFINE_VEC_OP_SCALAR_INTEGRAL_ASSIGN(|)
    DEFINE_VEC_OP_SCALAR_INTEGRAL_ASSIGN(^)
    DEFINE_VEC_OP_SCALAR_INTEGRAL_ASSIGN(>>)
    DEFINE_VEC_OP_SCALAR_INTEGRAL_ASSIGN(<<)

    // vector
#define DEFINE_VEC_OP_VECTOR_ASSIGN(OP)                                                            \
  template <typename OtherVecT, typename VecT = Derived, bool IsAssignable = is_access_lref<VecT>, \
            enable_if_all<                                                                         \
                std::is_convertible_v<typename VecT::value_type, typename OtherVecT::value_type>,  \
                IsAssignable> = 0>                                                                 \
  constexpr Derived& operator OP##=(VecInterface<OtherVecT> const& o) noexcept {                   \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                               \
    using R = math::op_result_t<value_type, typename OtherVecT::value_type>;                       \
    for (index_type i = 0; i != extent; ++i) this->val(i) = (R)this->val(i) OP((R)o.val(i));       \
    return static_cast<Derived&>(*this);                                                           \
  }
    DEFINE_VEC_OP_VECTOR_ASSIGN(+)
    DEFINE_VEC_OP_VECTOR_ASSIGN(-)
    DEFINE_VEC_OP_VECTOR_ASSIGN(*)
    DEFINE_VEC_OP_VECTOR_ASSIGN(/)

#define DEFINE_VEC_OP_VECTOR_INTEGRAL_ASSIGN(OP)                                                   \
  template <typename OtherVecT, typename VecT = Derived, bool IsAssignable = is_access_lref<VecT>, \
            enable_if_all<std::is_integral_v<typename VecT::value_type>,                           \
                          std::is_integral_v<typename OtherVecT::value_type>, IsAssignable> = 0>   \
  constexpr Derived& operator OP##=(VecInterface<OtherVecT> const& o) noexcept {                   \
    DECLARE_VEC_INTERFACE_ATTRIBUTES                                                               \
    using R = math::op_result_t<value_type, typename OtherVecT::value_type>;                       \
    for (index_type i = 0; i != extent; ++i) this->val(i) = (R)this->val(i) OP((R)o.val(i));       \
    return static_cast<Derived&>(*this);                                                           \
  }
    DEFINE_VEC_OP_VECTOR_INTEGRAL_ASSIGN(&)
    DEFINE_VEC_OP_VECTOR_INTEGRAL_ASSIGN(|)
    DEFINE_VEC_OP_VECTOR_INTEGRAL_ASSIGN(^)
    DEFINE_VEC_OP_VECTOR_INTEGRAL_ASSIGN(>>)
    DEFINE_VEC_OP_VECTOR_INTEGRAL_ASSIGN(<<)

    ///
    /// linalg
    ///
    // member func version
    template <typename OtherVecT, typename VecT = Derived,
              enable_if_all<std::is_arithmetic_v<typename VecT::value_type>,
                            std::is_arithmetic_v<typename OtherVecT::value_type>> = 0>
    constexpr auto dot(VecInterface<OtherVecT> const& rhs) const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      using R = math::op_result_t<value_type, typename OtherVecT::value_type>;
      R res{0};
      for (index_type i = 0; i != extent; ++i) res += this->val(i) * rhs.val(i);
      return res;
    }
    template <typename OtherVecT, typename VecT = Derived,
              enable_if_all<std::is_arithmetic_v<typename VecT::value_type>,
                            std::is_arithmetic_v<typename OtherVecT::value_type>, VecT::dim == 1,
                            OtherVecT::dim == 1, VecT::extent == 3, OtherVecT::extent == 3> = 0>
    constexpr auto cross(VecInterface<OtherVecT> const& rhs) const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      using R = math::op_result_t<value_type, typename OtherVecT::value_type>;
      typename Derived::template variant_vec<R, extents> res{};  // extents type follows 'lhs'
      res.val(0) = this->val(1) * rhs.val(2) - this->val(2) * rhs.val(1);
      res.val(1) = this->val(2) * rhs.val(0) - this->val(0) * rhs.val(2);
      res.val(2) = this->val(0) * rhs.val(1) - this->val(1) * rhs.val(0);
      return res;
    }
    // friend version
    template <typename OtherVecT, typename VecT = Derived,
              enable_if_all<std::is_arithmetic_v<typename VecT::value_type>,
                            std::is_arithmetic_v<typename OtherVecT::value_type>> = 0>
    friend constexpr auto dot(VecInterface const& lhs,
                              VecInterface<OtherVecT> const& rhs) noexcept {
      return lhs.dot(rhs);
    }
    template <typename OtherVecT, typename VecT = Derived,
              enable_if_all<std::is_arithmetic_v<typename VecT::value_type>,
                            std::is_arithmetic_v<typename OtherVecT::value_type>, VecT::dim == 1,
                            OtherVecT::dim == 1, VecT::extent == 3, OtherVecT::extent == 3> = 0>
    friend constexpr auto cross(VecInterface const& lhs,
                                VecInterface<OtherVecT> const& rhs) noexcept {
      return lhs.cross(rhs);
    }

    ///
    /// compare
    ///
    template <typename OtherVecT, typename VecT = Derived,
              enable_if_t<is_same_v<typename OtherVecT::extents, typename VecT::extents>> = 0>
    friend constexpr bool operator==(VecInterface const& lhs,
                                     VecInterface<OtherVecT> const& rhs) noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      for (index_type i = 0; i != extent; ++i)
        if (lhs.val(i) != rhs.val(i)) return false;
      return true;
    }
    template <typename OtherVecT, typename VecT = Derived,
              enable_if_t<is_same_v<typename OtherVecT::extents, typename VecT::extents>> = 0>
    friend constexpr bool operator!=(VecInterface const& lhs,
                                     VecInterface<OtherVecT> const& rhs) noexcept {
      return !(lhs == rhs);
    }

  protected:
    constexpr auto do_data() noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      return (value_type*)nullptr;
    }
    constexpr auto do_data() volatile noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      return (volatile value_type*)nullptr;
    }
    constexpr auto do_data() const noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      return (const value_type*)nullptr;
    }
    constexpr auto do_data() const volatile noexcept {
      DECLARE_VEC_INTERFACE_ATTRIBUTES
      return (const volatile value_type*)nullptr;
    }
  };

}  // namespace zs