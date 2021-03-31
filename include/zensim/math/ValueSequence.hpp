#ifndef __VALUE_SEQUENCE_HPP_
#define __VALUE_SEQUENCE_HPP_
#include <utility>

#include "Vec.h"

namespace zs {

  template <typename Tn, typename> struct integrals_impl;
  template <typename Tn, std::size_t N> struct integrals;

  template <typename Tn, std::size_t... Is> struct integrals_impl<Tn, std::index_sequence<Is...>>
      : vec<Tn, sizeof...(Is)> {
    static constexpr std::size_t N = sizeof...(Is);
    constexpr integrals_impl(const vec<Tn, N> &seqIn) noexcept : vec<Tn, N>{seqIn} {}

    constexpr Tn extent(Tn index) const noexcept { return (*this)(index); }
    constexpr Tn offset(const vec<Tn, N> &indices) const noexcept {
      return (... + (indices(Is) * (*this)(Is)));
    }
    constexpr Tn prod() const noexcept { return (... * (*this)(Is)); }
    constexpr Tn suffixProd(int i) const noexcept {
      Tn res{1};
      for (int d = i; d < N; ++d) res *= (*this)(d);
      return res;
    }
    constexpr integrals<Tn, N> seqExclSfxProd() const noexcept {
      return integrals<Tn, N>{vec<Tn, N>{suffixProd(Is + 1)...}};
    }
    template <typename Func> constexpr integrals<Tn, N> transform(Func func) const noexcept {
      return integrals<Tn, N>{vec<Tn, N>{func((*this)(Is))...}};
    }
  };

  template <typename Tn, std::size_t N> struct integrals
      : integrals_impl<Tn, std::make_index_sequence<N>> {
    using base_t = integrals_impl<Tn, std::make_index_sequence<N>>;
    constexpr integrals(const vec<Tn, N> &seq) noexcept : base_t{seq} {}
    constexpr integrals(vec<Tn, N> &&seq) noexcept : base_t{seq} {}
  };

}  // namespace zs

#endif