#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <type_traits>
#include <utility>

#include "VecInterface.hpp"
#include "zensim/math/MathUtils.h"
#include "zensim/meta/Meta.h"
#include "zensim/meta/Relationship.h"
#include "zensim/meta/Sequence.h"
#include "zensim/tpls/gcem/gcem.hpp"
#include "zensim/types/Tuple.h"

namespace zs {

  /// declarations
  template <typename T, typename Extents> struct vec_view;
  template <typename T, typename Extents> struct vec_impl;
#if 0
template <typename T, auto... Ns>
using vec =
    vec_impl<T,
             std::integer_sequence<std::common_type_t<decltype(Ns)...>, Ns...>>;
#else
  template <typename T, auto... Ns> using vec = vec_impl<T, std::integer_sequence<int, Ns...>>;
#endif
  template <typename T, typename Tn, Tn... Ns> using vec_t
      = vec_impl<T, std::integer_sequence<Tn, Ns...>>;

  template <typename T> struct is_vec : std::false_type {};
  template <typename T, auto... Ns> struct is_vec<vec<T, Ns...>> : std::true_type {};
  template <typename... Ts> struct is_vec<vec_view<Ts...>> : std::true_type {};
  template <typename... Ts> struct is_vec<vec_impl<Ts...>> : std::true_type {};

  /// vec without lifetime managing
  template <typename T, typename Tn, Tn... Ns> struct vec_view<T, std::integer_sequence<Tn, Ns...>>
      : indexer<Tn, Ns...> {
    using indexer_type = indexer<Tn, Ns...>;
    using indexer_type::dim;
    using indexer_type::extent;
    using value_type = T;
    using indexer_type::offset;
    using typename indexer_type::extents;
    using typename indexer_type::index_type;

    constexpr vec_view() = delete;
    constexpr explicit vec_view(T *ptr) : _data{ptr} {}

    /// random access
    // ()
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr T &operator()(Args &&...args) noexcept {
      return _data[indexer_type::offset(std::forward<Args>(args)...)];
    }
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr const T &operator()(Args &&...args) const noexcept {
      return _data[indexer_type::offset(std::forward<Args>(args)...)];
    }
    // []
    template <typename Index,
              typename R
              = vec_view<T, gather_t<typename gen_seq<dim - 1>::template arithmetic<1>, extents>>,
              Tn d = dim, enable_if_t<(d > 1)> = 0>
    constexpr R operator[](Index &&index) noexcept {
      return R{_data + indexer_type::offset(std::forward<Index>(index))};
    }
    template <typename Index,
              typename R
              = vec_view<std::add_const_t<T>,
                         gather_t<typename gen_seq<dim - 1>::template arithmetic<1>, extents>>,
              Tn d = dim, enable_if_t<(d > 1)> = 0>
    constexpr R operator[](Index index) const noexcept {
      return R{_data + indexer_type::offset(std::forward<Index>(index))};
    }
    template <typename Index, Tn d = dim, enable_if_t<d == 1> = 0>
    constexpr T &operator[](Index index) noexcept {
      return _data[std::forward<Index>(index)];
    }
    template <typename Index, Tn d = dim, enable_if_t<d == 1> = 0>
    constexpr const T &operator[](Index index) const noexcept {
      return _data[std::forward<Index>(index)];
    }
    template <typename Index> constexpr T &val(Index index) noexcept {
      return _data[std::forward<Index>(index)];
    }
    template <typename Index> constexpr const T &val(Index index) const noexcept {
      return _data[std::forward<Index>(index)];
    }

  private:
    T *_data;
  };

  /// vec
  template <typename T, typename Tn, Tn... Ns> struct vec_impl<T, std::integer_sequence<Tn, Ns...>>
      : VecInterface<vec_impl<T, std::integer_sequence<Tn, Ns...>>> {
    // static_assert(std::is_trivial<T>::value,
    //              "Vec element type is not trivial!\n");
    using indexer_type = indexer<Tn, Ns...>;
    static constexpr auto dim = sizeof...(Ns);
    static constexpr auto extent = (Ns * ...);
    using value_type = T;
    using index_type = Tn;
    using extents = std::integer_sequence<Tn, Ns...>;

    template <typename OtherT, typename IndicesT> using variant_vec = vec_impl<OtherT, IndicesT>;

    T _data[extent];

  public:
    /// expose internal
    constexpr auto data() noexcept -> T * { return _data; }
    constexpr auto data() volatile noexcept -> volatile T * { return (volatile T *)_data; }
    constexpr auto data() const noexcept -> const T * { return _data; }

    /// think this does not break rule of five
    constexpr vec_impl() = default;
    template <typename... Ts, enable_if_all<(sizeof...(Ts) <= extent),
                                            (std::is_convertible_v<Ts, value_type> && ...)> = 0>
    constexpr vec_impl(Ts &&...ts) noexcept : _data{(value_type)ts...} {}
    /// https://github.com/kokkos/kokkos/issues/177
    constexpr vec_impl &operator=(const vec_impl &o) = default;
#if 0
    constexpr volatile vec_impl &operator=(const vec_impl &o) volatile {
      for (Tn i = 0; i != extent; ++i) data()[i] = o.data()[i];
      return *this;
    }
#endif
    template <typename... Args, std::size_t... Is, enable_if_t<sizeof...(Args) == extent> = 0>
    static constexpr vec_impl from_tuple(const std::tuple<Args...> &tup, index_seq<Is...>) {
      vec_impl ret{};
      ((void)(ret.data()[Is] = std::get<Is>(tup)), ...);
      return ret;
    }
    template <typename... Args, enable_if_t<sizeof...(Args) == extent> = 0>
    static constexpr vec_impl from_tuple(const std::tuple<Args...> &tup) {
      return from_tuple(tup, std::index_sequence_for<Args...>{});
    }
    template <typename... Args, enable_if_t<sizeof...(Args) == extent> = 0>
    constexpr vec_impl &operator=(const std::tuple<Args...> &tup) {
      *this = from_tuple(tup);
      return *this;
    }
    template <typename... Args, std::size_t... Is, enable_if_t<sizeof...(Args) == extent> = 0>
    static constexpr vec_impl from_tuple(const zs::tuple<Args...> &tup, index_seq<Is...>) {
      vec_impl ret{};
      ((void)(ret.data()[Is] = zs::get<Is>(tup)), ...);
      return ret;
    }
    template <typename... Args, enable_if_t<sizeof...(Args) == extent> = 0>
    static constexpr vec_impl from_tuple(const zs::tuple<Args...> &tup) {
      return from_tuple(tup, std::index_sequence_for<Args...>{});
    }

    static constexpr vec_impl from_array(const std::array<T, extent> &arr) noexcept {
      vec_impl r{};
      for (Tn i = 0; i != extent; ++i) r.val(i) = arr[i];
      return r;
    }
    constexpr std::array<T, extent> to_array() const noexcept {
      std::array<T, extent> r{};
      for (Tn i = 0; i != extent; ++i) r[i] = _data[i];
      return r;
    }
    static constexpr vec_impl uniform(T v) noexcept {
      vec_impl r{};
      for (Tn i = 0; i != extent; ++i) r.val(i) = v;
      return r;
    }
    static constexpr vec_impl zeros() noexcept { return uniform(0); }
    template <int d = dim, Tn n = select_indexed_value<0, Ns...>::value,
              enable_if_all<d == 2, n * n == extent> = 0>
    static constexpr vec_impl identity() noexcept {
      vec_impl r = zeros();
      for (Tn i = 0; i != n; ++i) r(i, i) = (value_type)1;
      return r;
    }
    constexpr void set(T val) noexcept {
      for (Tn idx = 0; idx != extent; ++idx) _data[idx] = val;
    }
    /// random access
    // ()
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr T &operator()(Args &&...args) noexcept {
      return _data[indexer_type::offset(std::forward<Args>(args)...)];
    }
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr const T &operator()(Args &&...args) const noexcept {
      return _data[indexer_type::offset(std::forward<Args>(args)...)];
    }
    // []
    template <typename Index,
              typename R
              = vec_view<T, gather_t<typename gen_seq<dim - 1>::template arithmetic<1>, extents>>,
              Tn d = dim, enable_if_t<(d > 1)> = 0>
    constexpr R operator[](Index &&index) noexcept {
      return R{_data + indexer_type::offset(std::forward<Index>(index))};
    }
    template <typename Index,
              typename R
              = vec_view<std::add_const_t<T>,
                         gather_t<typename gen_seq<dim - 1>::template arithmetic<1>, extents>>,
              Tn d = dim, enable_if_t<(d > 1)> = 0>
    constexpr R operator[](Index index) const noexcept {
      return R{_data + indexer_type::offset(std::forward<Index>(index))};
    }
    template <typename Index, Tn d = dim, enable_if_t<d == 1> = 0>
    constexpr T &operator[](Index index) noexcept {
      return _data[std::forward<Index>(index)];
    }
    template <typename Index, Tn d = dim, enable_if_t<d == 1> = 0>
    constexpr const T &operator[](Index index) const noexcept {
      return _data[std::forward<Index>(index)];
    }
    template <typename Index> constexpr T &do_val(Index index) noexcept {
      return _data[std::forward<Index>(index)];
    }
    template <typename Index> constexpr const T &do_val(Index index) const noexcept {
      return _data[std::forward<Index>(index)];
    }
    ///
    template <typename TT> constexpr auto cast() const noexcept {
      vec_impl<TT, extents> r{};
      for (Tn idx = 0; idx != extent; ++idx) r.val(idx) = _data[idx];
      return r;
    }
    template <typename TT> constexpr operator vec_impl<TT, extents>() const noexcept {
      vec_impl<TT, extents> r{};
      for (Tn idx = 0; idx != extent; ++idx) r.val(idx) = _data[idx];
      return r;
    }
    /// compare
    template <typename TT>
    constexpr bool operator==(const vec_impl<TT, extents> &o) const noexcept {
      for (Tn i = 0; i != extent; ++i)
        if (_data[i] != o.val(i)) return false;
      return true;
    }
    template <typename TT>
    constexpr bool operator!=(const vec_impl<TT, extents> &&o) const noexcept {
      for (Tn i = 0; i != extent; ++i)
        if (_data[i] == o.val(i)) return false;
      return true;
    }

    /// linalg
    template <typename TT> constexpr auto dot(vec_impl<TT, extents> const &o) const noexcept {
      using R = math::op_result_t<T, TT>;
      R res{0};
      for (Tn i = 0; i != extent; ++i) res += _data[i] * o.val(i);
      return res;
    }
    template <typename TT, std::size_t d = dim, std::size_t ext = extent,
              enable_if_all<d == 1, ext == 3> = 0>
    constexpr auto cross(const vec_impl<TT, extents> &o) const noexcept {
      using R = math::op_result_t<T, TT>;
      vec_impl<R, extents> res{0};
      res.val(0) = _data[1] * o.val(2) - _data[2] * o.val(1);
      res.val(1) = _data[2] * o.val(0) - _data[0] * o.val(2);
      res.val(2) = _data[0] * o.val(1) - _data[1] * o.val(0);
      return res;
    }
    template <std::size_t d = dim, std::size_t ext = extent, enable_if_all<d == 1, ext == 3> = 0>
    constexpr vec_impl orthogonal() const noexcept {
      T x = gcem::abs(do_val(0));
      T y = gcem::abs(do_val(1));
      T z = gcem::abs(do_val(2));
      vec_impl other = x < y ? (x < z ? vec_impl{1, 0, 0} : vec_impl{0, 0, 1})
                             : (y < z ? vec_impl{0, 1, 0} : vec_impl{0, 0, 1});
      return cross(other);
    }
    template <std::size_t d = dim, enable_if_t<d == 2> = 0>
    constexpr auto transpose() const noexcept {
      constexpr auto N0 = select_indexed_value<0, Ns...>::value;
      constexpr auto N1 = select_indexed_value<1, Ns...>::value;
      using extentsT = std::integer_sequence<Tn, N1, N0>;
      vec_impl<T, extentsT> r{};
      for (Tn i = 0; i != N0; ++i)
        for (Tn j = 0; j != N1; ++j) r(j, i) = (*this)(i, j);
      return r;
    }
    constexpr T prod() const noexcept {
      T res{1};
      for (Tn i = 0; i != extent; ++i) res *= _data[i];
      return res;
    }
    constexpr T sum() const noexcept {
      T res{0};
      for (Tn i = 0; i != extent; ++i) res += _data[i];
      return res;
    }
    constexpr T l2NormSqr() const noexcept {
      T res{0};
      for (Tn i = 0; i != extent; ++i) res += _data[i] * _data[i];
      return res;
    }
    constexpr T infNormSqr() const noexcept {
      T res{0};
      for (Tn i = 0; i != extent; ++i)
        if (T sqr = _data[i] * _data[i]; sqr > res) res = sqr;
      return res;
    }
    static constexpr T sqrtNewtonRaphson(T x, T curr, T prev) noexcept {
      return curr == prev ? curr : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
    }
    constexpr T length() const noexcept {
      T sqrNorm = l2NormSqr();
      // return sqrtNewtonRaphson(sqrNorm, sqrNorm, (T)0);
      return gcem::sqrt(sqrNorm);
    }
    constexpr T norm() const noexcept {
      T sqrNorm = l2NormSqr();
      // return sqrtNewtonRaphson(sqrNorm, sqrNorm, (T)0);
      return gcem::sqrt(sqrNorm);
    }
    constexpr vec_impl normalized() const noexcept { return (*this) / length(); }
    constexpr vec_impl abs() const noexcept {
      vec_impl r{};
      for (Tn i = 0; i != extent; ++i) r.val(i) = _data[i] > 0 ? _data[i] : -_data[i];
      return r;
    }
    constexpr T max() const noexcept {
      T res{_data[0]};
      for (Tn i = 1; i != extent; ++i)
        if (_data[i] > res) res = _data[i];
      return res;
    }
  };

  /// make vec
  template <typename... Args, enable_if_all<(!is_std_tuple<remove_cvref_t<Args>>::value, ...)> = 0>
  constexpr auto make_vec(Args &&...args) noexcept {
    using Tn = math::op_result_t<remove_cvref_t<Args>...>;
    return vec<Tn, sizeof...(Args)>{FWD(args)...};
  }
  /// make vec from std tuple
  template <typename T, typename... Ts, std::size_t... Is>
  constexpr vec<T, (sizeof...(Ts))> make_vec_impl(const std::tuple<Ts...> &tup,
                                                  index_seq<Is...>) noexcept {
    return vec<T, (sizeof...(Ts))>{std::get<Is>(tup)...};
  }
  template <typename T, typename... Ts>
  constexpr auto make_vec(const std::tuple<Ts...> &tup) noexcept {
    return make_vec_impl<T>(tup, std::index_sequence_for<Ts...>{});
  }
  /// make vec from zs tuple
  template <typename T, typename... Ts, std::size_t... Is>
  constexpr vec<T, (sizeof...(Ts))> make_vec_impl(const tuple<Ts...> &tup,
                                                  index_seq<Is...>) noexcept {
    return vec<T, (sizeof...(Ts))>{get<Is>(tup)...};
  }
  template <typename T, typename... Ts> constexpr auto make_vec(const tuple<Ts...> &tup) noexcept {
    return make_vec_impl<T>(tup, std::index_sequence_for<Ts...>{});
  }

  template <typename T, typename Tn, Tn... Ns>
  constexpr auto vectorize(const vec_impl<T, std::integer_sequence<Tn, Ns...>> &v) noexcept {
    constexpr Tn s = (Ns * ...);
    vec_impl<T, std::integer_sequence<Tn, s>> ret{};
    for (Tn i = 0; i != s; ++i) ret(i) = v(i);
    return ret;
  }

  /// affine map = linear map + translation matrix+(0, 0, 1) point(vec+{1})
  /// vector(vec+{0}) homogeneous coordinates

  template <typename... Args> constexpr auto make_array(Args &&...args) {
    return std::array<math::op_result_t<remove_cvref_t<Args>...>, sizeof...(Args)>{FWD(args)...};
  }
  template <typename RetT, typename... Args> constexpr auto make_array(Args &&...args) {
    return std::array<RetT, sizeof...(Args)>{FWD(args)...};
  }

  template <typename Index, typename T, int dim>
  constexpr auto world_to_index(const vec<T, dim> &pos, float dxinv, Index offset = 0) {
    vec<Index, dim> coord{};
    for (int d = 0; d < dim; ++d) coord[d] = lower_trunc(pos[d] * dxinv + 0.5f) + offset;
    return coord;
  }

  template <typename Tn, int dim>
  constexpr auto unpack_coord(const vec<Tn, dim> &id, Tn side_length) {
    using T = std::make_signed_t<Tn>;
    auto bid = id;
    for (int d = 0; d != dim; ++d) bid[d] += (id[d] < 0 ? ((T)1 - (T)side_length) : 0);
    bid = bid / side_length;
    return std::make_tuple(bid, id - bid * side_length);
  }
  template <int dim, typename Tn, typename Ti>
  constexpr auto linear_to_coord(Tn offset, Ti sideLength) {
    using T = math::op_result_t<Tn, Ti>;
    vec<Tn, dim> ret{};
    for (int d = dim - 1; d != -1 && offset; --d, offset = (T)offset / (T)sideLength)
      ret[d] = (T)offset % (T)sideLength;
    return ret;
  }

}  // namespace zs

#include "MatrixUtils.inl"