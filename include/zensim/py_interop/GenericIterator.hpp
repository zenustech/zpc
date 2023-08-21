#pragma once
#include "zensim/math/Vec.h"
#include "zensim/types/Property.h"

namespace zs {

  /// @note assume the tile size is a power of two.
  /// @note assume either scalar or zs::vec (where T is its value_type, dim is its extent)
  template <typename T, auto... Ns> struct aosoa_iterator
      : IteratorInterface<aosoa_iterator<T, Ns...>> {
    using primitive_type = T;
    using iter_value_type = remove_cvref_t<primitive_type>;
    static constexpr auto extent = (Ns * ...);
    static constexpr bool is_const_structure = is_const_v<primitive_type>;
    static constexpr bool is_scalar_access = (extent == 1);

    using size_type = u32;
    using difference_type = make_signed_t<size_type>;

    constexpr aosoa_iterator() noexcept = default;
    ~aosoa_iterator() = default;

    /// aos
    constexpr aosoa_iterator(wrapv<layout_e::aos>, T* ptr, size_type id) noexcept
        : base{ptr}, idx{id}, numTileBits{0}, tileMask{0}, numChns{extent} {}
    /// soa requires knowledge of 'tileSize',
    /// for practical reasons, discard the support of this variant
    /// aosoa
    constexpr aosoa_iterator(wrapv<layout_e::aosoa>, T* ptr, size_type id, size_type tileSize,
                             size_type chnOffset, size_type numChns) noexcept
        : base{ptr + chnOffset * tileSize},
          idx{0},
          numTileBits{bit_count(tileSize)},
          tileMask{tileSize - (size_type)1},
          numChns{numChns} {}

    constexpr decltype(auto) dereference() const noexcept {
      if constexpr (is_scalar_access) {
        /// @note (i >> numTileBits) full tiles
        /// @note each tile covers (numChns << numTileBits) entries
        return *(base + ((((idx >> numTileBits) * numChns) << numTileBits) | (idx & tileMask)));
      } else {
        using PtrT = conditional_t<is_const_structure, const iter_value_type*, iter_value_type*>;
        /// @note when T is non-const, remove the constness of base
        using RetT = vec<PtrT, Ns...>;
        RetT ret{};
        auto ptr
            = (PtrT)base + ((((idx >> numTileBits) * numChns) << numTileBits) | (idx & tileMask));
        for (int d = 0; d != extent; ++d, ptr += (tileMask + 1)) ret._data[d] = ptr;
        return ret;
      }
    }
    constexpr bool equal_to(const aosoa_iterator& it) const noexcept {
      return it.base == base && it.idx == idx;
    }
    constexpr void advance(difference_type offset) noexcept { idx += offset; }
    constexpr difference_type distance_to(const aosoa_iterator& it) const noexcept {
      return it.idx - idx;
    }

    /// @note for aosoa, base already embeds channel offset
    T* base;  /// @note base pointer already embeds the channel offset
    size_type idx, numTileBits, tileMask, numChns;
  };

}  // namespace zs

extern "C" {

#define ZS_DECLARE_GENERIC_ITERATOR_TYPE(T, n)                       \
  template struct zs::aosoa_iterator<T, n>;                          \
  typedef zs::aosoa_iterator<T, n> aosoa_iter_##T##_##n;             \
  template struct zs::aosoa_iterator<const T, n>;                    \
  typedef zs::aosoa_iterator<const T, n> aosoa_iter_const_##T##_##n; \
  template struct zs::aosoa_iterator<T, n, n>;                       \
  typedef zs::aosoa_iterator<T, n, n> aosoa_iter_##T##_##n##_##n;    \
  template struct zs::aosoa_iterator<const T, n, n>;                 \
  typedef zs::aosoa_iterator<const T, n, n> aosoa_iter_const_##T##_##n##_##n;

ZS_DECLARE_GENERIC_ITERATOR_TYPE(float, 1)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(float, 2)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(float, 3)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(float, 4)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(float, 6)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(float, 9)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(float, 12)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(double, 1)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(double, 2)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(double, 3)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(double, 4)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(double, 6)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(double, 9)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(double, 12)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(int, 1)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(int, 2)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(int, 3)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(int, 4)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(int, 6)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(int, 9)
ZS_DECLARE_GENERIC_ITERATOR_TYPE(int, 12)
}
