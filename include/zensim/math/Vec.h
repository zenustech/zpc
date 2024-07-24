#pragma once

// #  include <array>
// #  include <cmath>
// #  include <functional>
// #  include <type_traits>
// #  include <utility>

#include "VecInterface.hpp"
#include "zensim/ZpcImplPattern.hpp"
#include "zensim/ZpcIterator.hpp"
#include "zensim/ZpcMathUtils.hpp"
#include "zensim/ZpcTuple.hpp"

namespace std {
#if defined(ZS_PLATFORM_OSX)
  inline namespace __1 {
#endif

    template <typename, zs::size_t> struct array;

#if defined(ZS_PLATFORM_OSX)
  }
#endif
}  // namespace std

namespace zs {

  /// declarations
  template <typename T, typename Extents> struct vec_view;
  template <typename T, typename Extents> struct vec_impl;

  template <typename T, auto... Ns> using vec = vec_impl<T, integer_sequence<int, Ns...>>;
  template <typename T, typename Tn, Tn... Ns> using vec_t
      = vec_impl<T, integer_sequence<Tn, Ns...>>;

  /// vec without lifetime managing
  template <typename T, typename Tn, Tn... Ns> struct vec_view<T, integer_sequence<Tn, Ns...>>
      : Mixin<vec_view<T, integer_sequence<Tn, Ns...>>, VecInterface, Visitee> {
    static constexpr bool is_pointer_structure = is_pointer_v<T>;
    using base_t = VecInterface<vec_view<T, integer_sequence<Tn, Ns...>>>;
    // essential defs for any VecInterface
    using primitive_type = T;
    using value_type = remove_pointer_t<T>;
    using index_type = Tn;
    using indexer_type = indexer<index_type, Ns...>;
    using extents = integer_sequence<index_type, Ns...>;

    SUPPLEMENT_VEC_STATIC_ATTRIBUTES

    template <typename OtherT, typename IndicesT> using variant_vec = vec_impl<OtherT, IndicesT>;

    template <bool IsPtrStruct = is_pointer_structure, enable_if_t<!IsPtrStruct> = 0>
    constexpr vec_view() noexcept = delete;
    constexpr vec_view() noexcept = default;
    ~vec_view() = default;
    constexpr vec_view(const vec_view &) = delete;             // prevents accidental copy of view
    constexpr vec_view &operator=(const vec_view &) = delete;  // prevents accidental copy of view
    template <bool IsPtrStruct = is_pointer_structure, enable_if_t<!IsPtrStruct> = 0>
    constexpr explicit vec_view(T *ptr) : _data{ptr} {}
    template <bool IsPtrStruct = is_pointer_structure, enable_if_t<IsPtrStruct> = 0>
    constexpr explicit vec_view(T ptrs[]) : _data{} {
      for (index_type i = 0; i != extent; ++i) _data[i] = ptrs[i];
    }

    constexpr explicit operator variant_vec<value_type, extents>() const noexcept {
      variant_vec<value_type, extents> r{};
      for (index_type i = 0; i != extent; ++i) r.val(i) = base_t::val(i);
      return r;
    }

    /// random access
    // ()
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr decltype(auto) operator()(Args... args) noexcept {
      return base_t::val(indexer_type::offset(args...));
    }
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr decltype(auto) operator()(Args... args) const noexcept {
      return base_t::val(indexer_type::offset(args...));
    }
    // []
    constexpr decltype(auto) operator[](index_type index) noexcept {
      if constexpr (dim == 1) {
        return base_t::val(index);
      } else {
        using R = vec_view<T, gather_t<typename build_seq<dim - 1>::template arithmetic<1>, extents>>;
        const auto st = indexer_type::offset(index);
        if constexpr (is_pointer_structure) {
          R ret{};
          for (index_type i = 0; i != R::extent; ++i) ret._data[i] = base_t::data(st + i);
          return ret;
        } else
          return R{base_t::data(st)};
      }
    }
    constexpr decltype(auto) operator[](index_type index) const noexcept {
      if constexpr (dim == 1) {
        return base_t::val(index);
      } else {
        using TT = conditional_t<is_pointer_structure, add_pointer_t<add_const_t<value_type>>,
                                 add_const_t<value_type>>;
        using R
            = vec_view<TT, gather_t<typename build_seq<dim - 1>::template arithmetic<1>, extents>>;
        const auto st = indexer_type::offset(index);
        if constexpr (is_pointer_structure) {
          R ret{};
          for (index_type i = 0; i != R::extent; ++i) ret._data[i] = base_t::data(st + i);
          return ret;
        } else {
          return R{base_t::data(st)};
        }
      }
    }
    constexpr auto do_data(index_type i) noexcept {
      if constexpr (is_pointer_structure)
        return _data[i];
      else
        return _data + i;
    }
    constexpr auto do_data(index_type i) const volatile noexcept {
      if constexpr (is_pointer_structure)
        return _data[i];
      else
        return _data + i;
    }
    constexpr auto do_data(index_type i) volatile noexcept {
      if constexpr (is_pointer_structure)
        return _data[i];
      else
        return _data + i;
    }
    constexpr auto do_data(index_type i) const noexcept {
      if constexpr (is_pointer_structure)
        return _data[i];
      else
        return _data + i;
    }

    conditional_t<is_pointer_structure, T[extent], T *> _data;
  };

  /// vec
  template <typename T, typename Tn, Tn... Ns> struct vec_impl<T, integer_sequence<Tn, Ns...>>
      : Mixin<vec_impl<T, integer_sequence<Tn, Ns...>>, VecInterface, Visitee> {
    // static_assert(std::is_trivial<T>::value,
    //              "Vec element type is not trivial!\n");
    static constexpr bool is_pointer_structure = is_pointer_v<T>;
    using base_t = VecInterface<vec_impl<T, integer_sequence<Tn, Ns...>>>;
    using primitive_type = T;
    using value_type = remove_pointer_t<T>;
    using index_type = Tn;
    using indexer_type = indexer<index_type, Ns...>;
    using extents = integer_sequence<index_type, Ns...>;

    SUPPLEMENT_VEC_STATIC_ATTRIBUTES

    template <typename OtherT, typename IndicesT> using variant_vec = vec_impl<OtherT, IndicesT>;

    T _data[extent];  // empty-initialized for both arithmetic types and pointer types

  public:
    /// expose internal
    constexpr auto do_data(index_type i) noexcept {
      if constexpr (is_pointer_structure)
        return _data[i];
      else
        return _data + i;
    }
    constexpr auto do_data(index_type i) const volatile noexcept {
      if constexpr (is_pointer_structure)
        return _data[i];
      else
        return _data + i;
    }
    constexpr auto do_data(index_type i) volatile noexcept {
      if constexpr (is_pointer_structure)
        return _data[i];
      else
        return _data + i;
    }
    constexpr auto do_data(index_type i) const noexcept {
      if constexpr (is_pointer_structure)
        return _data[i];
      else
        return _data + i;
    }

    /// think this does not break rule of five
    constexpr vec_impl() noexcept = default;
    constexpr vec_impl(const vec_impl &) noexcept = default;
    constexpr vec_impl(vec_impl &&) noexcept = default;
    constexpr vec_impl &operator=(const vec_impl &) & noexcept = default;
    constexpr vec_impl &operator=(vec_impl &&) & noexcept = default;
    template <typename... Ts, bool IsPtrStruct = is_pointer_structure,
              enable_if_all<!IsPtrStruct, (sizeof...(Ts) <= extent),
                            (is_convertible_v<remove_cvref_t<Ts>, value_type> && ...)>
              = 0>
    constexpr vec_impl(Ts... ts) noexcept : _data{(value_type)zs::move(ts)...} {}
    template <typename... Ts, bool IsPtrStruct = is_pointer_structure,
              enable_if_all<IsPtrStruct, (sizeof...(Ts) == extent),
                            ((alignof(Ts) == alignof(value_type)) && ...)>
              = 0>
    constexpr vec_impl(Ts &...ts) noexcept : _data{((T) const_cast<RM_CVREF_T(ts) *>(&ts))...} {}
    /// https://github.com/kokkos/kokkos/issues/177
#if 0
    constexpr volatile vec_impl &operator=(const vec_impl &o) volatile {
      for (Tn i = 0; i != extent; ++i) data()[i] = o.data()[i];
      return *this;
    }
#endif
    template <template <typename...> class TupT, typename... Args, size_t... Is,
              bool IsPtrStruct = is_pointer_structure,
              enable_if_all<sizeof...(Args) == extent, !IsPtrStruct> = 0>
    static constexpr vec_impl from_tuple(const TupT<Args...> &tup, index_sequence<Is...>) {
      vec_impl ret{};
      ((void)(ret.val(Is) = get<Is>(tup)), ...);  // ADL
      return ret;
    }
    template <template <typename...> class TupT, typename... Args,
              bool IsPtrStruct = is_pointer_structure,
              enable_if_all<sizeof...(Args) == extent, !IsPtrStruct> = 0>
    static constexpr auto from_tuple(const TupT<Args...> &tup)
        -> decltype(get<0>(tup), declval<vec_impl>()) {
      return from_tuple(tup, index_sequence_for<Args...>{});
    }
    template <template <typename...> class TupT>
    static constexpr vec_impl from_tuple(const TupT<> &tup) {
      return vec_impl{};
    }
    template <typename... Args, bool IsPtrStruct = is_pointer_structure,
              enable_if_all<sizeof...(Args) == extent, !IsPtrStruct> = 0>
    constexpr vec_impl &operator=(const tuple<Args...> &tup) {
      *this = from_tuple(tup);
      return *this;
    }

    template <template <typename, zs::size_t> class ArrayT, bool IsPtrStruct = is_pointer_structure,
              enable_if_t<!IsPtrStruct> = 0>
    static constexpr vec_impl from_array(const ArrayT<value_type, extent> &arr) noexcept {
      vec_impl r{};
      for (Tn i = 0; i != extent; ++i) r.val(i) = arr[i];
      return r;
    }
    template <template <typename, zs::size_t> class ArrayT = std::array,
              bool IsPtrStruct = is_pointer_structure, enable_if_t<!IsPtrStruct> = 0>
    constexpr ArrayT<value_type, extent> to_array() const noexcept {
      ArrayT<value_type, extent> r{};
      for (Tn i = 0; i != extent; ++i) r[i] = base_t::val(i);
      return r;
    }
    /// random access
    // ()
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr decltype(auto) operator()(Args... args) noexcept {
      return base_t::val(indexer_type::offset(args...));
    }
    template <typename... Args, enable_if_t<sizeof...(Args) <= dim> = 0>
    constexpr decltype(auto) operator()(Args... args) const noexcept {
      return base_t::val(indexer_type::offset(args...));
    }
    // []
    template <typename Index, enable_if_t<is_integral_v<Index>> = 0>
    constexpr decltype(auto) operator[](Index index) noexcept {
      if constexpr (dim == 1) {
        return base_t::val(index);
      } else {
        using R = vec_view<T, gather_t<typename build_seq<dim - 1>::template arithmetic<1>, extents>>;
        auto offset = indexer_type::offset(index);
        if constexpr (is_pointer_structure) {
          R ret{};
          for (index_type i = 0; i != R::extent; ++i) ret.data(i) = base_t::data(offset + i);
          return ret;
        } else
          return R{base_t::data(offset)};
      }
    }
    template <typename Index, enable_if_t<is_integral_v<Index>> = 0>
    constexpr decltype(auto) operator[](Index index) const noexcept {
      if constexpr (dim == 1) {
        return base_t::val(index);
      } else {
        using TT = conditional_t<is_pointer_structure, add_pointer_t<add_const_t<value_type>>,
                                 add_const_t<value_type>>;
        using R
            = vec_view<TT, gather_t<typename build_seq<dim - 1>::template arithmetic<1>, extents>>;
        auto offset = indexer_type::offset(index);
        if constexpr (is_pointer_structure) {
          R ret{};
          for (index_type i = 0; i != R::extent; ++i) ret.data(i) = base_t::data(offset + i);
          return ret;
        } else
          return R{base_t::data(offset)};
      }
    }
    template <typename TT,
              enable_if_all<(alignof(value_type) >= alignof(TT)), sizeof(value_type) == sizeof(TT)>
              = 0>
    constexpr auto reinterpret_bits(wrapt<TT> = {}) const noexcept {
#if 0
      variant_vec<TT, extents> r{};
      std::memcpy(r.data(), data(), sizeof(TT) * extent);
      return r;
#else
      return base_t::reinterpret_bits(wrapt<TT>{});
#endif
    }
    ///
    template <typename TT> constexpr explicit operator vec_impl<TT, extents>() const noexcept {
      vec_impl<TT, extents> r{};
      for (Tn idx = 0; idx != extent; ++idx) r.val(idx) = base_t::val(idx);
      return r;
    }
  };

#if ZS_ENABLE_SERIALIZATION
  template <typename S, typename T, typename Tn, Tn... Ns,
            enable_if_t<(is_arithmetic_v<T> || is_enum_v<T>)> = 0>
  void serialize(S &s, vec_impl<T, integer_sequence<Tn, Ns...>> &v) {
    auto r = zs::range(v.data(), v.data() + v.extent);
    s.template container<sizeof(T)>(r);
  }
#endif

  /// make vec
  template <typename... Args, enable_if_all<((is_fundamental_v<remove_cvref_t<Args>>), ...)> = 0>
  constexpr auto make_vec(Args... args) noexcept {
    using Tn = math::op_result_t<remove_cvref_t<Args>...>;
    return vec<Tn, sizeof...(Args)>{zs::move(args)...};
  }
  /// make vec from std/zs tuple
  template <typename T, template <typename...> class TupT, typename... Ts, size_t... Is>
  constexpr vec<T, (sizeof...(Ts))> make_vec_impl(const TupT<Ts...> &tup,
                                                  index_sequence<Is...>) noexcept {
    return vec<T, (sizeof...(Ts))>{get<Is>(tup)...};
  }
  template <typename T, template <typename...> class TupT, typename... Ts>
  constexpr auto make_vec(const TupT<Ts...> &tup) noexcept {
    return make_vec_impl<T>(tup, index_sequence_for<Ts...>{});
  }

  /// affine map = linear map + translation matrix+(0, 0, 1) point(vec+{1})
  /// vector(vec+{0}) homogeneous coordinates

#if 0
  template <typename... Args> constexpr auto make_array(Args ...args) {
    return std::array<math::op_result_t<remove_cvref_t<Args>...>, sizeof...(Args)>{args...};
  }
  template <typename RetT, typename... Args> constexpr auto make_array(Args ...args) {
    return std::array<RetT, sizeof...(Args)>{args...};
  }
#endif

  template <typename Tn, int dim>
  constexpr auto unpack_coord(const vec<Tn, dim> &id, Tn side_length) {
    using T = zs::make_signed_t<Tn>;
    auto bid = id;
    for (int d = 0; d != dim; ++d) bid[d] += (id[d] < 0 ? ((T)1 - (T)side_length) : 0);
    bid = bid / side_length;
    return zs::make_tuple(bid, id - bid * side_length);
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

namespace zs {

  template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == 3> = 0>
  constexpr auto cross_matrix(const VecInterface<VecT> &v) noexcept {
    // skew_symmetric matrix
    // cross-product matrix
    typename VecT::template variant_vec<typename VecT::value_type,
                                        integer_sequence<typename VecT::index_type, 3, 3>>
        m{};
    m(0, 0) = m(1, 1) = m(2, 2) = 0;
    m(0, 1) = -v(2);
    m(0, 2) = v(1);
    m(1, 0) = v(2);
    m(1, 2) = -v(0);
    m(2, 0) = -v(1);
    m(2, 1) = v(0);
    return m;
  }

  template <typename VecTM, bool ColMajorOrder = true, enable_if_all<VecTM::dim == 2> = 0>
  constexpr auto vectorize(const VecInterface<VecTM> &m, wrapv<ColMajorOrder> = {}) noexcept {
    constexpr auto nrows = VecTM::template range_t<0>::value;
    constexpr auto ncols = VecTM::template range_t<1>::value;
    typename VecTM::template variant_vec<
        typename VecTM::value_type, integer_sequence<typename VecTM::index_type, nrows * ncols>>
        r{};
    if constexpr (ColMajorOrder) {
      for (typename VecTM::index_type j = 0, no = 0; j != ncols; ++j)
        for (typename VecTM::index_type i = 0; i != nrows; ++i) r(no++) = m(i, j);
    } else {
      for (typename VecTM::index_type i = 0, no = 0; i != nrows; ++i)
        for (typename VecTM::index_type j = 0; j != ncols; ++j) r(no++) = m(i, j);
    }
    return r;
  }

  template <typename VecTM, enable_if_all<VecTM::dim == 2> = 0>
  constexpr auto row(const VecInterface<VecTM> &m, typename VecTM::index_type i) noexcept {
    constexpr auto ncols = VecTM::template range_t<1>::value;
    typename VecTM::template variant_vec<typename VecTM::value_type,
                                         integer_sequence<typename VecTM::index_type, ncols>>
        r{};
    for (typename VecTM::index_type j = 0; j != ncols; ++j) r.val(j) = m(i, j);
    return r;
  }
  template <typename VecTM, enable_if_all<VecTM::dim == 2> = 0>
  constexpr auto col(const VecInterface<VecTM> &m, typename VecTM::index_type j) noexcept {
    constexpr auto nrows = VecTM::template range_t<0>::value;
    typename VecTM::template variant_vec<typename VecTM::value_type,
                                         integer_sequence<typename VecTM::index_type, nrows>>
        c{};
    for (typename VecTM::index_type i = 0; i != nrows; ++i) c.val(i) = m(i, j);
    return c;
  }
  template <typename VecTM, bool ColumnMajor = true, enable_if_all<VecTM::dim == 2> = 0>
  constexpr auto flatten(const VecInterface<VecTM> &m, wrapv<ColumnMajor> = {}) noexcept {
    constexpr auto nrows = VecTM::template range_t<0>::value;
    constexpr auto ncols = VecTM::template range_t<1>::value;
    using index_type = typename VecTM::index_type;
    typename VecTM::template variant_vec<typename VecTM::value_type,
                                         integer_sequence<index_type, nrows * ncols>>
        r{};
    if constexpr (ColumnMajor) {
      for (index_type offset = 0, j = 0; j != ncols; ++j)
        for (index_type i = 0; i != nrows; ++i) r(offset++) = m(i, j);
    } else {
      for (index_type offset = 0, i = 0; i != nrows; ++i)
        for (index_type j = 0; j != ncols; ++j) r(offset++) = m(i, j);
    }
    return r;
  }

  template <typename VecTM, enable_if_all<VecTM::dim == 2, VecTM::template range_t<0>::value
                                                               == VecTM::template range_t<1>::value>
                            = 0>
  constexpr auto trace(const VecInterface<VecTM> &m) noexcept {
    constexpr auto n = VecTM::template range_t<0>::value;
    typename VecTM::value_type r{};
    for (typename VecTM::index_type i = 0; i != n; ++i) r += m(i, i);
    return r;
  }

  /// vector-vector product
  template <typename VecTA, typename VecTB, enable_if_all<VecTA::dim == 1, VecTB::dim == 1> = 0>
  constexpr auto dyadic_prod(const VecInterface<VecTA> &col,
                             const VecInterface<VecTB> &row) noexcept {
    using R = math::op_result_t<typename VecTA::value_type, typename VecTB::value_type>;
    using index_type = typename VecTA::index_type;
    constexpr auto nrows = VecTA::extent;
    constexpr auto ncols = VecTB::extent;
    typename VecTA::template variant_vec<R, integer_sequence<index_type, nrows, ncols>> m{};
    for (index_type i = 0; i != nrows; ++i)
      for (index_type j = 0; j != ncols; ++j) m(i, j) = col.val(i) * row.val(j);
    return m;
  }

  namespace detail {
    template <int i0, int i1, typename VecT, enable_if_t<VecT::dim == 2> = 0>
    constexpr auto det2(const VecInterface<VecT> &A) noexcept {
      return A(i0, 0) * A(i1, 1) - A(i1, 0) * A(i0, 1);
    }
    template <int i0, int i1, int i2, typename VecT, typename T,
              enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == 4,
                            VecT::template range_t<1>::value == 4>
              = 0>
    constexpr auto det3(const VecInterface<VecT> &A, const T &d0, const T &d1,
                        const T &d2) noexcept {
      return A(i0, 2) * d0 + (-A(i1, 2) * d1 + A(i2, 2) * d2);
    }
    template <int i, int j, typename VecT,
              enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == 3,
                            VecT::template range_t<1>::value == 3>
              = 0>
    constexpr typename VecT::value_type cofactor(const VecInterface<VecT> &A) noexcept {
      constexpr int i1 = (i + 1) % 3;
      constexpr int i2 = (i + 2) % 3;
      constexpr int j1 = (j + 1) % 3;
      constexpr int j2 = (j + 2) % 3;
      return A(i1, j1) * A(i2, j2) - A(i1, j2) * A(i2, j1);
    }
    template <int i1, int i2, int i3, int j1, int j2, int j3, typename VecT,
              enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == 4,
                            VecT::template range_t<1>::value == 4>
              = 0>
    constexpr typename VecT::value_type det3(const VecInterface<VecT> &A) noexcept {
      return A(i1, j1) * (A(i2, j2) * A(i3, j3) - A(i2, j3) * A(i3, j2));
    }
    template <int i, int j, typename VecT,
              enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == 4,
                            VecT::template range_t<1>::value == 4>
              = 0>
    constexpr typename VecT::value_type cofactor(const VecInterface<VecT> &A) noexcept {
      constexpr int i1 = (i + 1) % 4;
      constexpr int i2 = (i + 2) % 4;
      constexpr int i3 = (i + 3) % 4;
      constexpr int j1 = (j + 1) % 4;
      constexpr int j2 = (j + 2) % 4;
      constexpr int j3 = (j + 3) % 4;
      return det3<i1, i2, i3, j1, j2, j3>(A) + det3<i2, i3, i1, j1, j2, j3>(A)
             + det3<i3, i1, i2, j1, j2, j3>(A);
    }
  }  // namespace detail

  template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == 1,
                                         VecT::template range_t<1>::value == 1>
                           = 0>
  constexpr auto determinant(const VecInterface<VecT> &A) noexcept {
    return A(0, 0);
  }
  template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == 2,
                                         VecT::template range_t<1>::value == 2>
                           = 0>
  constexpr auto determinant(const VecInterface<VecT> &A) noexcept {
    return A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1);
  }
  template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == 3,
                                         VecT::template range_t<1>::value == 3>
                           = 0>
  constexpr auto determinant(const VecInterface<VecT> &A) noexcept {
    return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1))
           - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0))
           + A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
  }
  template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == 4,
                                         VecT::template range_t<1>::value == 4>
                           = 0>
  constexpr auto determinant(const VecInterface<VecT> &A) noexcept {
    auto d2_01 = detail::det2<0, 1>(A);
    auto d2_02 = detail::det2<0, 2>(A);
    auto d2_03 = detail::det2<0, 3>(A);
    auto d2_12 = detail::det2<1, 2>(A);
    auto d2_13 = detail::det2<1, 3>(A);
    auto d2_23 = detail::det2<2, 3>(A);
    auto d3_0 = detail::det3<1, 2, 3>(A, d2_23, d2_13, d2_12);
    auto d3_1 = detail::det3<0, 2, 3>(A, d2_23, d2_03, d2_02);
    auto d3_2 = detail::det3<0, 1, 3>(A, d2_13, d2_03, d2_01);
    auto d3_3 = detail::det3<0, 1, 2>(A, d2_12, d2_02, d2_01);
    return -A(0, 3) * d3_0 + A(1, 3) * d3_1 + -A(2, 3) * d3_2 + A(3, 3) * d3_3;
  }
  template <typename VecT,
            enable_if_all<VecT::dim == 2,
                          VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                          (VecT::template range_t<0>::value > 4)>
            = 0>
  constexpr auto determinant(const VecInterface<VecT> &A) noexcept {
    using Ti = typename VecT::index_type;
    using T = typename VecT::value_type;
    constexpr auto dim = VecT::template range_t<0>::value;
    using SubMatT = typename VecT::template variant_vec<T, integer_sequence<Ti, dim - 1, dim - 1>>;
    T det = 0;
    SubMatT m{};
    for (Ti j = 0; j != dim; ++j) {
      for (Ti r = 1; r != dim; ++r) {
        int cOffset = 0;
        for (Ti c = 0; c != dim; ++c) {
          if (c == j) {
            cOffset = 1;
            continue;
          }
          m(r - 1, c - cOffset) = A(r, c);
        }
      }
      det += (j & 1 ? -1 : 1) * A(0, j) * determinant(m);
    }
    return det;
  }

  template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == 1,
                                         VecT::template range_t<1>::value == 1,
                                         is_floating_point_v<typename VecT::value_type>>
                           = 0>
  constexpr auto inverse(const VecInterface<VecT> &A) noexcept {
    using T = typename VecT::value_type;
    auto ret = VecT::zeros();
    ret(0, 0) = (T)1 / A(0, 0);
    return ret;
  }
  template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == 2,
                                         VecT::template range_t<1>::value == 2,
                                         is_floating_point_v<typename VecT::value_type>>
                           = 0>
  constexpr auto inverse(const VecInterface<VecT> &A) noexcept {
    using T = typename VecT::value_type;
    auto ret = VecT::zeros();
    auto invdet = (T)1 / determinant(A);
    ret(0, 0) = A(1, 1) * invdet;
    ret(1, 0) = -A(1, 0) * invdet;
    ret(0, 1) = -A(0, 1) * invdet;
    ret(1, 1) = A(0, 0) * invdet;
    return ret;
  }
  template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == 3,
                                         VecT::template range_t<1>::value == 3,
                                         is_floating_point_v<typename VecT::value_type>>
                           = 0>
  constexpr auto inverse(const VecInterface<VecT> &A) noexcept {
    using T = typename VecT::value_type;
    auto ret = VecT::zeros();
    ret(0, 0) = detail::cofactor<0, 0>(A);
    ret(0, 1) = detail::cofactor<1, 0>(A);
    ret(0, 2) = detail::cofactor<2, 0>(A);
    const T invdet = (T)1 / (ret(0, 0) * A(0, 0) + ret(0, 1) * A(1, 0) + ret(0, 2) * A(2, 0));
    T c01 = detail::cofactor<0, 1>(A) * invdet;
    T c11 = detail::cofactor<1, 1>(A) * invdet;
    T c02 = detail::cofactor<0, 2>(A) * invdet;
    ret(1, 2) = detail::cofactor<2, 1>(A) * invdet;
    ret(2, 1) = detail::cofactor<1, 2>(A) * invdet;
    ret(2, 2) = detail::cofactor<2, 2>(A) * invdet;
    ret(1, 0) = c01;
    ret(1, 1) = c11;
    ret(2, 0) = c02;
    ret(0, 0) *= invdet;
    ret(0, 1) *= invdet;
    ret(0, 2) *= invdet;
    return ret;
  }
  template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == 4,
                                         VecT::template range_t<1>::value == 4,
                                         is_floating_point_v<typename VecT::value_type>>
                           = 0>
  constexpr auto inverse(const VecInterface<VecT> &A) noexcept {
    // using T = typename VecT::value_type;
    auto ret = VecT::zeros();
    ret(0, 0) = detail::cofactor<0, 0>(A);
    ret(1, 0) = -detail::cofactor<0, 1>(A);
    ret(2, 0) = detail::cofactor<0, 2>(A);
    ret(3, 0) = -detail::cofactor<0, 3>(A);

    ret(0, 2) = detail::cofactor<2, 0>(A);
    ret(1, 2) = -detail::cofactor<2, 1>(A);
    ret(2, 2) = detail::cofactor<2, 2>(A);
    ret(3, 2) = -detail::cofactor<2, 3>(A);

    ret(0, 1) = -detail::cofactor<1, 0>(A);
    ret(1, 1) = detail::cofactor<1, 1>(A);
    ret(2, 1) = -detail::cofactor<1, 2>(A);
    ret(3, 1) = detail::cofactor<1, 3>(A);

    ret(0, 3) = -detail::cofactor<3, 0>(A);
    ret(1, 3) = detail::cofactor<3, 1>(A);
    ret(2, 3) = -detail::cofactor<3, 2>(A);
    ret(3, 3) = detail::cofactor<3, 3>(A);
    return ret
           / (A(0, 0) * ret(0, 0) + A(1, 0) * ret(0, 1) + A(2, 0) * ret(0, 2)
              + A(3, 0) * ret(0, 3));
  }

  template <typename VecTM, typename VecTV,
            enable_if_all<VecTM::dim == 2, VecTV::dim == 1,
                          VecTM::template range_t<1>::value == VecTV::template range_t<0>::value>
            = 0>
  constexpr auto diag_mul(const VecInterface<VecTM> &A, const VecInterface<VecTV> &diag) noexcept {
    using R = math::op_result_t<typename VecTM::value_type, typename VecTV::value_type>;
    typename VecTM::template variant_vec<R, typename VecTM::extents> r{};
    for (typename VecTM::index_type i = 0; i != VecTM::template range_t<0>::value; ++i)
      for (typename VecTM::index_type j = 0; j != VecTM::template range_t<1>::value; ++j)
        r(i, j) = A(i, j) * diag(j);
    return r;
  }
  template <typename VecTV, typename VecTM,
            enable_if_all<VecTV::dim == 1, VecTM::dim == 2,
                          VecTM::template range_t<0>::value == VecTV::template range_t<0>::value>
            = 0>
  constexpr auto diag_mul(const VecInterface<VecTV> &diag, const VecInterface<VecTM> &A) noexcept {
    using R = math::op_result_t<typename VecTM::value_type, typename VecTV::value_type>;
    typename VecTM::template variant_vec<R, typename VecTM::extents> r{};
    for (typename VecTM::index_type i = 0; i != VecTM::template range_t<0>::value; ++i)
      for (typename VecTM::index_type j = 0; j != VecTM::template range_t<1>::value; ++j)
        r(i, j) = A(i, j) * diag(i);
    return r;
  }

  /// affine transform
  template <
      typename VecTM, typename VecTV,
      enable_if_all<VecTM::dim == 2, VecTV::dim == 1,
                    VecTM::template range_t<1>::value == VecTV::template range_t<0>::value + 1>
      = 0>
  constexpr auto operator*(const VecInterface<VecTM> &A, const VecInterface<VecTV> &x) noexcept {
    using R = math::op_result_t<typename VecTM::value_type, typename VecTV::value_type>;
    using index_type = typename VecTM::index_type;
    constexpr auto nrows_m_1 = VecTM::template range_t<0>::value - 1;
    constexpr auto ncols = VecTM::template range_t<1>::value;
    constexpr auto ncols_m_1 = ncols - 1;
    typename VecTM::template variant_vec<R, integer_sequence<index_type, nrows_m_1>> r{};
    for (index_type i = 0; i != nrows_m_1; ++i) {
      r(i) = (R)0;
      for (index_type j = 0; j != ncols; ++j) r(i) += (j == ncols_m_1 ? A(i, j) : A(i, j) * x(j));
    }
    return r;
  }
  template <
      typename VecTV, typename VecTM,
      enable_if_all<VecTV::dim == 1, VecTM::dim == 2,
                    VecTM::template range_t<0>::value == VecTV::template range_t<0>::value + 1>
      = 0>
  constexpr auto operator*(const VecInterface<VecTV> &x, const VecInterface<VecTM> &A) noexcept {
    using R = math::op_result_t<typename VecTM::value_type, typename VecTV::value_type>;
    using index_type = typename VecTV::index_type;
    constexpr auto nrows = VecTM::template range_t<0>::value;
    constexpr auto nrows_m_1 = nrows - 1;
    constexpr auto ncols_m_1 = VecTM::template range_t<1>::value - 1;
    typename VecTV::template variant_vec<R, integer_sequence<index_type, ncols_m_1>> r{};
    for (index_type j = 0; j != ncols_m_1; ++j) {
      r(j) = (R)0;
      for (index_type i = 0; i != nrows; ++i) r(j) += (i == nrows_m_1 ? A(i, j) : A(i, j) * x(i));
    }
    return r;
  }
}  // namespace zs