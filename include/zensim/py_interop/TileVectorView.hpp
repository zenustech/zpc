#pragma once
#include "zensim/ZpcMeta.hpp"
#include "zensim/ZpcTuple.hpp"
#include "zensim/math/Vec.h"
#include "zensim/types/SmallVector.hpp"

namespace zs {

  template <typename T_, size_t Length, bool WithinTile = false>
  struct TileVectorViewLite {  // T may be const
    using value_type = remove_const_t<T_>;
    using size_type = size_t;
    using channel_counter_type = int;
    static constexpr size_t lane_width = Length;

    static constexpr bool is_const_structure = is_const<T_>::value;

    constexpr TileVectorViewLite() noexcept = default;
    TileVectorViewLite(T_* const v, const channel_counter_type nchns) noexcept
        : _vector{v}, _numChannels{nchns} {}

    constexpr channel_counter_type numChannels() const noexcept { return _numChannels; }

    /// @brief [chn, no]
    template <bool V = is_const_structure, typename TT = value_type,
              enable_if_all<!V, sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            alignof(TT) == alignof(value_type)>
              = 0>
    constexpr add_lvalue_reference_t<TT> operator()(const channel_counter_type chn,
                                                    const size_type i, wrapt<TT> = {}) noexcept {
      if constexpr (WithinTile) {
        return *((TT*)_vector + (chn * lane_width + i));
      } else {
        return *((TT*)_vector
                 + ((i / lane_width * _numChannels + chn) * lane_width + i % lane_width));
      }
    }
    template <typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr TT operator()(const channel_counter_type chn, const size_type i,
                            wrapt<TT> = {}) const noexcept {
      if constexpr (WithinTile) {
        return *((const TT*)_vector + (chn * lane_width + i));
      } else {
        return *((const TT*)_vector
                 + ((i / lane_width * _numChannels + chn) * lane_width + i % lane_width));
      }
    }
    /// @brief [chn, tileNo, localNo]
    template <typename TT = value_type, bool V = is_const_structure, bool InTile = WithinTile,
              enable_if_all<!V, !InTile, sizeof(TT) == sizeof(value_type),
                            is_same_v<TT, remove_cvref_t<TT>>, (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr add_lvalue_reference_t<TT> operator()(const channel_counter_type chn,
                                                    const size_type tileNo, const size_type localNo,
                                                    wrapt<TT> = {}) noexcept {
      return *((TT*)_vector + ((tileNo * _numChannels + chn) * lane_width + localNo));
    }
    template <typename TT = value_type, bool InTile = WithinTile,
              enable_if_all<!InTile, sizeof(TT) == sizeof(value_type),
                            is_same_v<TT, remove_cvref_t<TT>>, (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr TT operator()(const channel_counter_type chn, const size_type tileNo,
                            const size_type localNo, wrapt<TT> = {}) const noexcept {
      return *((const TT*)_vector + ((tileNo * _numChannels + chn) * lane_width + localNo));
    }
    /// @brief tile
    template <bool V = is_const_structure, bool InTile = WithinTile, enable_if_all<!V, !InTile> = 0>
    constexpr auto tile(const size_type tileid) noexcept {
      return TileVectorViewLite<T_, lane_width, true>{_vector + tileid * lane_width * _numChannels,
                                                      _numChannels};
    }
    template <bool InTile = WithinTile, enable_if_t<!InTile> = 0>
    constexpr auto tile(const size_type tileid) const noexcept {
      return TileVectorViewLite<T_, lane_width, true>{_vector + tileid * lane_width * _numChannels,
                                                      _numChannels};
    }
    /// @brief pack [chn, no]
    // use dim_c<Ns...> for the first parameter
    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto pack(value_seq<Ns...>, channel_counter_type chn, const size_type i,
                        wrapt<TT> = {}) const noexcept {
      using RetT = vec<TT, Ns...>;
      RetT ret{};
      const TT* ptr = nullptr;
      if constexpr (WithinTile) {
        ptr = (const TT*)_vector + (chn * lane_width + i);
      } else {
        ptr = (const TT*)_vector
              + ((i / lane_width * _numChannels + chn) * lane_width + (i % lane_width));
      }
      for (channel_counter_type d = 0; d != RetT::extent; ++d, ptr += lane_width) ret.val(d) = *ptr;
      return ret;
    }
    /// @brief tuple [chn, no]
    template <size_t... Is, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto tuple_impl(const channel_counter_type chnOffset, const size_type i,
                              index_sequence<Is...>, wrapt<TT>) const noexcept {
      if constexpr (WithinTile)
        return zs::tie(*((TT*)_vector
                         + ((size_type)chnOffset + (size_type)Is) * (size_type)lane_width + i)...);
      else {
        size_type a{}, b{};
        a = i / lane_width * _numChannels;
        b = i % lane_width;
        return zs::tie(*((TT*)_vector
                         + (a + ((size_type)chnOffset + (size_type)Is)) * (size_type)lane_width
                         + b)...);
      }
    }

    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto tuple(value_seq<Ns...>, const channel_counter_type chn, const size_type i,
                         wrapt<TT> = {}) const noexcept {
      return tuple_impl(chn, i, make_index_sequence<(Ns * ...)>{}, wrapt<TT>{});
    }

    T_* _vector{nullptr};
    channel_counter_type _numChannels{0};
  };

  template <typename T_, size_t Length, bool WithinTile = false> struct TileVectorNamedViewLite
      : TileVectorViewLite<T_, Length, WithinTile> {
    using base_t = TileVectorViewLite<T_, Length, WithinTile>;
    using base_t::_numChannels;
    using base_t::_vector;
    using value_type = typename base_t::value_type;
    using size_type = typename base_t::size_type;
    using channel_counter_type = typename base_t::channel_counter_type;
    static constexpr auto lane_width = base_t::lane_width;

    static constexpr bool is_const_structure = base_t::is_const_structure;

    TileVectorNamedViewLite() noexcept = default;
    TileVectorNamedViewLite(T_* const v, const channel_counter_type nchns,
                            const SmallString* tagNames, const channel_counter_type* tagOffsets,
                            const channel_counter_type* tagSizes,
                            const channel_counter_type N) noexcept
        : base_t{v, nchns},
          _N{N},
          _tagNames{tagNames},
          _tagOffsets{tagOffsets},
          _tagSizes{tagSizes} {}

    constexpr auto getPropertyNames() const noexcept { return _tagNames; }
    constexpr auto getPropertyOffsets() const noexcept { return _tagOffsets; }
    constexpr auto getPropertySizes() const noexcept { return _tagSizes; }
    constexpr auto numProperties() const noexcept { return _N; }
    constexpr auto propertyIndex(const SmallString& propName) const noexcept {
      channel_counter_type i = 0;
      for (; i != _N; ++i)
        if (_tagNames[i] == propName) break;
      return i;
    }
    constexpr auto propertySize(const SmallString& propName) const noexcept {
      return getPropertySizes()[propertyIndex(propName)];
    }
    constexpr auto propertyOffset(const SmallString& propName) const noexcept {
      return getPropertyOffsets()[propertyIndex(propName)];
    }
    constexpr bool hasProperty(const SmallString& propName) const noexcept {
      return propertyIndex(propName) != _N;
    }

    using base_t::operator();
    using base_t::numChannels;
    using base_t::pack;
    using base_t::tuple;

    template <bool V = is_const_structure, typename TT = value_type,
              enable_if_all<!V, sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) <= alignof(value_type))>
              = 0>
    constexpr TT& operator()(const SmallString& propName, const channel_counter_type chn,
                             const size_type i, wrapt<TT> = {}) noexcept {
      auto propNo = propertyIndex(propName);
      return static_cast<base_t&>(*this)(_tagOffsets[propNo] + chn, i, wrapt<TT>{});
    }
    template <typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr TT operator()(const SmallString& propName, const channel_counter_type chn,
                            const size_type i, wrapt<TT> = {}) const noexcept {
      auto propNo = propertyIndex(propName);
      return static_cast<const base_t&>(*this)(_tagOffsets[propNo] + chn, i, wrapt<TT>{});
    }
    template <bool V = is_const_structure, typename TT = value_type,
              enable_if_all<!V, sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr TT& operator()(const SmallString& propName, const size_type i,
                             wrapt<TT> = {}) noexcept {
      auto propNo = propertyIndex(propName);
      return static_cast<base_t&>(*this)(_tagOffsets[propNo], i, wrapt<TT>{});
    }
    template <typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) <= alignof(value_type))>
              = 0>
    constexpr TT operator()(const SmallString& propName, const size_type i,
                            wrapt<TT> = {}) const noexcept {
      auto propNo = propertyIndex(propName);
      return static_cast<const base_t&>(*this)(_tagOffsets[propNo], i, wrapt<TT>{});
    }

    template <bool V = is_const_structure, bool InTile = WithinTile, enable_if_all<!V, !InTile> = 0>
    constexpr auto tile(const size_type tileid) noexcept {
      return TileVectorNamedViewLite<value_type, lane_width, true>{
          this->_vector + tileid * lane_width * base_t::_numChannels,
          base_t::_numChannels,
          _tagNames,
          _tagOffsets,
          _tagSizes,
          _N};
    }
    template <bool InTile = WithinTile, enable_if_t<!InTile> = 0>
    constexpr auto tile(const size_type tileid) const noexcept {
      return TileVectorNamedViewLite<value_type, lane_width, true>{
          this->_vector + tileid * lane_width * base_t::_numChannels,
          base_t::_numChannels,
          _tagNames,
          _tagOffsets,
          _tagSizes,
          _N};
    }

    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) <= alignof(value_type))>
              = 0>
    constexpr auto pack(value_seq<Ns...>, const SmallString& propName, const size_type i,
                        wrapt<TT> = {}) const noexcept {
      auto propNo = propertyIndex(propName);
      return static_cast<const base_t&>(*this).pack(dim_c<Ns...>, _tagOffsets[propNo], i,
                                                    wrapt<TT>{});
    }
    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) <= alignof(value_type))>
              = 0>
    constexpr auto pack(value_seq<Ns...>, const SmallString& propName,
                        const channel_counter_type chn, const size_type i,
                        wrapt<TT> = {}) const noexcept {
      auto propNo = propertyIndex(propName);
      return static_cast<const base_t&>(*this).pack(dim_c<Ns...>, _tagOffsets[propNo] + chn, i,
                                                    wrapt<TT>{});
    }

    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) <= alignof(value_type))>
              = 0>
    constexpr auto tuple(value_seq<Ns...>, const SmallString& propName, const size_type i,
                         wrapt<TT> = {}) const noexcept {
      auto propNo = propertyIndex(propName);
      return static_cast<const base_t&>(*this).tuple(dim_c<Ns...>, _tagOffsets[propNo], i,
                                                     wrapt<TT>{});
    }
    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) <= alignof(value_type))>
              = 0>
    constexpr auto tuple(value_seq<Ns...>, const SmallString& propName,
                         const channel_counter_type chn, const size_type i,
                         wrapt<TT> = {}) const noexcept {
      auto propNo = propertyIndex(propName);
      return static_cast<const base_t&>(*this).tuple(dim_c<Ns...>, _tagOffsets[propNo] + chn, i,
                                                     wrapt<TT>{});
    }

    const SmallString* _tagNames{nullptr};
    const channel_counter_type* _tagOffsets{nullptr};
    const channel_counter_type* _tagSizes{nullptr};
    channel_counter_type _N{0};
  };

}  // namespace zs