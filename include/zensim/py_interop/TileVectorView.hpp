#pragma once
#include "zensim/ZpcMeta.hpp"
#include "zensim/types/SmallVector.hpp"

namespace zs {

  template <typename T_, size_t Length> struct TileVectorViewLite {  // T may be const
    using value_type = T_;
    using size_type = size_t;
    using channel_counter_type = int;
    static constexpr size_t lane_width = Length;

    static constexpr bool is_const_structure = is_const<value_type>::value;

    TileVectorViewLite() noexcept = default;
    TileVectorViewLite(value_type* const v, const channel_counter_type nchns) noexcept
        : _vector{v}, _numChannels{nchns} {}

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr decltype(auto) operator[](size_type i) {
      return _vector[i];
    }
    constexpr auto operator[](size_type i) const { return _vector[i]; }

    value_type* _vector{nullptr};
    channel_counter_type _numChannels{0};
  };

  template <typename T_, size_t Length> struct TileVectorNamedViewLite
      : TileVectorViewLite<T_, Length> {
    using base_t = TileVectorViewLite<T_, Length>;
    using base_t::_numChannels;
    using base_t::_vector;
    using value_type = typename base_t::value_type;
    using size_type = typename base_t::size_type;
    using channel_counter_type = typename base_t::channel_counter_type;
    static constexpr size_t lane_width = base_t::lane_width;

    static constexpr bool is_const_structure = base_t::is_const_structure;

    TileVectorNamedViewLite() noexcept = default;
    TileVectorNamedViewLite(value_type* const v, const channel_counter_type nchns,
                            const SmallString* tagNames, const channel_counter_type* tagOffsets,
                            const channel_counter_type* tagSizes,
                            const channel_counter_type N) noexcept
        : base_t{v, nchns},
          _N{N},
          _tagNames{tagNames},
          _tagOffsets{tagOffsets},
          _tagSizes{tagSizes} {}

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr decltype(auto) operator[](size_type i) {
      return _vector[i];
    }
    constexpr auto operator[](size_type i) const { return _vector[i]; }

    channel_counter_type _N{0};
    const SmallString* _tagNames{nullptr};
    const channel_counter_type* _tagOffsets{nullptr};
    const channel_counter_type* _tagSizes{nullptr};
  };

}  // namespace zs