#pragma once
#include "zensim/math/Vec.h"
#include "zensim/types/Property.h"

namespace zs {

  /// @note assume either scalar or zs::vec (where T is its value_type, dim is its extent)
  template <typename T, int dim> struct aosoa_view {
    using value_type = remove_const_t<T>;
    using size_type = u32;
    constexpr aosoa_view() noexcept = default;
    ~aosoa_view() = default;

    /// aos
    constexpr aosoa_view(wrapv<layout_e::aos>, T* ptr, size_type nchns = dim) noexcept
        : base{ptr}, numTileBits{0}, tileMask{(size_type)0}, numChns{nchns} {}
    /// soa
    constexpr aosoa_view(wrapv<layout_e::soa>, T* ptr, size_type size, size_type chnOffset) noexcept
        : aosoa_view(wrapv<layout_e::aos>{}, ptr + size * chnOffset, (size_type)1) {}
    /// aosoa
    constexpr aosoa_view(wrapv<layout_e::aosoa>, T* ptr, size_type tileSize, size_type chnOffset,
                         size_type numChns) noexcept
        : base{ptr + chnOffset * tileSize},
          numTileBits{bit_count(tileSize)},
          tileMask{(size_type)(tileSize - 1)},
          numChns{numChns} {}

    constexpr auto operator()(size_type i) const noexcept {
      if constexpr (is_vec<value_type>::value) {
        ;
      } else {
        /// @note index[i] covers (i >> numTileBits) full tiles
        /// @note index[i] resides as the ei & tileMask] entry in its tile
        return *(base + ((((i >> numTileBits) * numChns) << numTileBits) | (i & tileMask)));
      }
    }

    /// @note for aosoa, base already embeds channel offset
    T* base;
    size_type numTileBits, tileMask, numChns;
  };

}  // namespace zs