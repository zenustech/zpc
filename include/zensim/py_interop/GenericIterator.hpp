#pragma once
#include "zensim/math/Vec.h"
#include "zensim/types/Property.h"

namespace zs {

  /// @note assume the tile size is a power of two.
  /// @note assume either scalar or zs::vec (where T is its value_type, dim is its extent)
  template <typename T, auto... Ns> struct aosoa_iterator;

  template <typename T, auto... Ns> struct aosoa_iterator_port {
    using size_type = u32;

    T* base;  /// @note base pointer already embeds the channel offset
    size_type idx, numTileBits, tileMask, numChns;
  };
  template <typename T, auto... Ns> struct aosoa_iterator
      : IteratorInterface<aosoa_iterator<T, Ns...>> {
    using primitive_type = T;
    using iter_value_type = remove_cvref_t<primitive_type>;
    using port_type = aosoa_iterator_port<T, Ns...>;
    static constexpr auto extent = (Ns * ...);
    static constexpr bool is_const_structure = is_const_v<primitive_type>;
    static constexpr bool is_scalar_access = (extent == 1);

    using size_type = typename port_type::size_type;
    using difference_type = make_signed_t<size_type>;

    using element_pointer_type
        = conditional_t<is_const_structure, const iter_value_type*, iter_value_type*>;
    using reference_type
        = conditional_t<is_scalar_access, decltype(*declval<element_pointer_type>()),
                        vec<element_pointer_type, Ns...>>;

    explicit constexpr aosoa_iterator(port_type it) noexcept
        : base{it.base},
          idx{it.idx},
          numTileBits{it.numTileBits},
          tileMask{it.tileMask},
          numChns{it.numChns} {}
    constexpr operator port_type() noexcept {
      port_type ret{};
      ret.base = base;
      ret.idx = idx;
      ret.numTileBits = numTileBits;
      ret.tileMask = tileMask;
      ret.numChns = numChns;
      return ret;
    }
    constexpr aosoa_iterator() noexcept = default;
    template <bool IsConst = is_const_structure, enable_if_t<IsConst> = 0>
    constexpr aosoa_iterator(const aosoa_iterator<iter_value_type, Ns...>& o) noexcept
        : base{o.base},
          idx{o.idx},
          numTileBits{o.numTileBits},
          tileMask{o.tileMask},
          numChns{o.numChns} {}
    template <bool IsConst = is_const_structure, enable_if_t<IsConst> = 0>
    constexpr aosoa_iterator& operator=(const aosoa_iterator<iter_value_type, Ns...>& o) noexcept {
      base = o.base;
      idx = o.idx;
      numTileBits = o.numTileBits;
      tileMask = o.tileMask;
      numChns = o.numChns;
      return *this;
    }

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
          idx{id},
          numTileBits{bit_count(tileSize)},
          tileMask{tileSize - (size_type)1},
          numChns{numChns} {}

    constexpr reference_type dereference() const noexcept {
      if constexpr (is_scalar_access) {
        /// @note (i >> numTileBits) full tiles
        /// @note each tile covers (numChns << numTileBits) entries
        return *((element_pointer_type)base
                 + ((((idx >> numTileBits) * numChns) << numTileBits) | (idx & tileMask)));
      } else {
        /// @note when T is non-const, remove the constness of base
        reference_type ret{};
        auto ptr = (element_pointer_type)base
                   + ((((idx >> numTileBits) * numChns) << numTileBits) | (idx & tileMask));
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

/// @note make sure these are POD types
#define ZS_DECLARE_GENERIC_ITERATOR_TYPE(T, n)                                               \
  template struct zs::aosoa_iterator_port<T, n>;                                             \
  template struct zs::aosoa_iterator<T, n>;                                                  \
  template struct zs::LegacyIterator<zs::aosoa_iterator<T, n>>;                              \
  typedef zs::aosoa_iterator_port<T, n> aosoa_iterator_port_##T##_##n;                       \
  typedef zs::aosoa_iterator<T, n> aosoa_iter_##T##_##n;                                     \
  typedef zs::LegacyIterator<zs::aosoa_iterator<T, n>> aosoa_iterator_##T##_##n;             \
  template struct zs::aosoa_iterator_port<const T, n>;                                       \
  template struct zs::aosoa_iterator<const T, n>;                                            \
  template struct zs::LegacyIterator<zs::aosoa_iterator<const T, n>>;                        \
  typedef zs::aosoa_iterator_port<const T, n> aosoa_iterator_port_const_##T##_##n;           \
  typedef zs::aosoa_iterator<const T, n> aosoa_iter_const_##T##_##n;                         \
  typedef zs::LegacyIterator<zs::aosoa_iterator<const T, n>> aosoa_iterator_const_##T##_##n; \
  template struct zs::aosoa_iterator_port<T, n, n>;                                          \
  template struct zs::aosoa_iterator<T, n, n>;                                               \
  template struct zs::LegacyIterator<zs::aosoa_iterator<T, n, n>>;                           \
  typedef zs::aosoa_iterator_port<T, n, n> aosoa_iterator_port_##T##_##n##_##n;              \
  typedef zs::aosoa_iterator<T, n, n> aosoa_iter_##T##_##n##_##n;                            \
  typedef zs::LegacyIterator<zs::aosoa_iterator<T, n, n>> aosoa_iterator_##T##_##n##_##n;    \
  template struct zs::aosoa_iterator_port<const T, n, n>;                                    \
  template struct zs::aosoa_iterator<const T, n, n>;                                         \
  template struct zs::LegacyIterator<zs::aosoa_iterator<const T, n, n>>;                     \
  typedef zs::aosoa_iterator_port<const T, n, n> aosoa_iterator_port_const_##T##_##n##_##n;  \
  typedef zs::aosoa_iterator<const T, n, n> aosoa_iter_const_##T##_##n##_##n;                \
  typedef zs::LegacyIterator<zs::aosoa_iterator<const T, n, n>>                              \
      aosoa_iterator_const_##T##_##n##_##n;

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
