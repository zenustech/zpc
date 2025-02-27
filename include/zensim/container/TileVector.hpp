#pragma once

#include "Vector.hpp"
#include "zensim/math/Vec.h"
#include "zensim/resource/Resource.h"
#include "zensim/types/Iterator.h"
#include "zensim/types/SmallVector.hpp"
#if ZS_ENABLE_SERIALIZATION
#  include "zensim/zpc_tpls/bitsery/traits/vector.h"
#endif

namespace zs {

  template <typename T_, size_t Length = 8, typename AllocatorT = ZSPmrAllocator<>>
  struct TileVector {
    static_assert(is_zs_allocator<AllocatorT>::value,
                  "TileVector only works with zspmrallocator for now.");
    static_assert(is_same_v<T_, remove_cvref_t<T_>>, "T is not cvref-unqualified type!");
    static_assert(std::is_default_constructible_v<T_> && std::is_trivially_copyable_v<T_>,
                  "element is not default-constructible or trivially-copyable!");

    using value_type = T_;
    using allocator_type = AllocatorT;
    using size_type = size_t;
    using difference_type = zs::make_signed_t<size_type>;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    // tile vector specific configs
    using channel_counter_type = int;
    static constexpr size_type lane_width = Length;

    static_assert(
        std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_swap::value,
        "allocator should propagate on copy, move and swap (for impl simplicity)!");

    static constexpr size_type count_tiles(size_type elementCount) noexcept {
      return (elementCount + lane_width - 1) / lane_width;
    }

    constexpr decltype(auto) memoryLocation() const noexcept { return _allocator.location; }
    constexpr ProcID devid() const noexcept { return memoryLocation().devid(); }
    constexpr memsrc_e memspace() const noexcept { return memoryLocation().memspace(); }
    decltype(auto) get_allocator() const noexcept { return _allocator; }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      if constexpr (is_virtual_zs_allocator<allocator_type>::value)
        return get_virtual_memory_source(mre, devid, (size_t)1 << (size_t)36, "STACK");
      else
        return get_memory_source(mre, devid);
    }
    pointer allocate(size_t bytes) {
      /// virtual memory way
      if constexpr (is_virtual_zs_allocator<allocator_type>::value) {
        _allocator.commit(0, bytes);
        return (pointer)_allocator.address(0);
      }
      /// conventional way
      else
        return (pointer)_allocator.allocate(bytes, alignof(value_type));
    }

    TileVector(const allocator_type &allocator, const std::vector<PropertyTag> &channelTags,
               size_type count = 0)
        : _allocator{allocator},
          _buffer{},
          _tags{channelTags},
          _size{count},
          _numChannels{numTotalChannels(channelTags)} {
      const auto N = numProperties();
      _buffer = Vector<value_type, allocator_type>{
          _allocator, count_tiles(count) * lane_width * static_cast<size_type>(_numChannels)};
      {
        auto tagNames = Vector<SmallString, allocator_type>{static_cast<size_t>(N)};
        auto tagSizes = Vector<channel_counter_type, allocator_type>{static_cast<size_t>(N)};
        auto tagOffsets = Vector<channel_counter_type, allocator_type>{static_cast<size_t>(N)};
        channel_counter_type curOffset = 0;
        for (auto &&[name, size, offset, src] : zip(tagNames, tagSizes, tagOffsets, channelTags)) {
          name = src.name;
          size = src.numChannels;
          offset = curOffset;
          curOffset += size;
        }
        _tagNames = tagNames.clone(_allocator);
        _tagSizes = tagSizes.clone(_allocator);
        _tagOffsets = tagOffsets.clone(_allocator);
      }
    }
    TileVector(const std::vector<PropertyTag> &channelTags, size_type count = 0,
               memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : TileVector{get_default_allocator(mre, devid), channelTags, count} {}
    TileVector(channel_counter_type numChns, size_type count = 0, memsrc_e mre = memsrc_e::host,
               ProcID devid = -1)
        : TileVector{get_default_allocator(mre, devid), {{"unnamed", numChns}}, count} {}
    TileVector(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : TileVector{get_default_allocator(mre, devid), {{"unnamed", 1}}, 0} {}

    ~TileVector() = default;

    pointer data() { return _buffer.data(); }
    const_pointer data() const { return _buffer.data(); }
    template <typename Dims = value_seq<1>, typename T = const value_type,
              enable_if_t<sizeof(T) == sizeof(value_type) && alignof(T) == alignof(value_type)> = 0>
    inline auto getVal(channel_counter_type chn = 0, size_type i = 0, Dims dims = {},
                       wrapt<T> = {}) const {
      auto ptr = data() + (i / lane_width * _numChannels + chn) * lane_width + i % lane_width;
      constexpr auto ext = static_value_extent<Dims>::value;
      if constexpr (ext == 1) {
        remove_cvref_t<T> ret{};
        Resource::copy(MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)&ret},
                       MemoryEntity{memoryLocation(), ptr}, sizeof(T));
        return ret;
      } else {
        using VS = decltype(value_seq{dims});
        vec_impl<remove_cvref_t<T>, typename VS::template to_iseq<int>> ret{};
        for (typename Dims::value_type d = 0; d != ext; ++d, ptr += lane_width)
          Resource::copy(MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)ret.data(d)},
                         MemoryEntity{memoryLocation(), ptr}, sizeof(T));
        return ret;
      }
    }
    template <typename VecT,
              enable_if_t<sizeof(typename VecT::value_type) == sizeof(value_type)
                          && alignof(typename VecT::value_type) == alignof(value_type)>
              = 0>
    inline void setVal(const VecInterface<VecT> &v, channel_counter_type chn = 0,
                       size_type i = 0) const {
      auto ptr = data() + (i / lane_width * _numChannels + chn) * lane_width + i % lane_width;

      for (typename VecT::value_type d = 0; d != VecT::extent; ++d, ptr += lane_width)
        Resource::copy(MemoryEntity{memoryLocation(), ptr},
                       MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)v.data(d)},
                       sizeof(value_type));
    }
    template <typename T,
              enable_if_t<sizeof(T) == sizeof(value_type) && alignof(T) == alignof(value_type)> = 0>
    inline void setVal(const T &v, channel_counter_type chn = 0, size_type i = 0) {
      auto ptr = data() + (i / lane_width * _numChannels + chn) * lane_width + i % lane_width;
      Resource::copy(MemoryEntity{memoryLocation(), ptr},
                     MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)&v}, sizeof(v));
    }

    static auto numTotalChannels(const std::vector<PropertyTag> &tags) {
      channel_counter_type cnt = 0;
      for (size_t i = 0; i != tags.size(); ++i) cnt += tags[i].numChannels;
      return cnt;
    }

    template <typename T = value_type, typename Dims = value_seq<1>, typename Pred = void>
    struct iterator_impl;

    /// @brief tilevector iterator specializations
    template <typename ValT, auto... Ns>
    struct iterator_impl<ValT, value_seq<Ns...>, enable_if_type<sizeof(value_type) == sizeof(ValT)>>
        : IteratorInterface<iterator_impl<ValT, value_seq<Ns...>,
                                          enable_if_type<sizeof(value_type) == sizeof(ValT)>>> {
      static_assert(!std::is_reference_v<ValT>,
                    "the access type of the iterator should not be a reference.");
      static constexpr auto extent = (Ns * ...);
      static_assert(extent > 0, "the access extent of the iterator should be positive.");

      using iter_value_type = remove_cvref_t<ValT>;
      static constexpr bool is_const_structure = is_const_v<ValT>;
      static constexpr bool is_native_value_type = is_same_v<value_type, iter_value_type>;
      static constexpr bool is_scalar_access = (extent == 1);

      constexpr iterator_impl(conditional_t<is_const_structure, const_pointer, pointer> base,
                              size_type idx, channel_counter_type chn,
                              channel_counter_type nchns) noexcept
          : _base{base}, _idx{idx}, _chn{chn}, _numChannels{nchns} {}

      constexpr decltype(auto) dereference() const {
        if constexpr (is_scalar_access)
          return *(
              (conditional_t<is_const_structure, const iter_value_type *, iter_value_type *>)_base
              + (_idx / lane_width * _numChannels + _chn) * lane_width + _idx % lane_width);
        else {
          using PtrT
              = conditional_t<is_const_structure, const iter_value_type *, iter_value_type *>;
          using RetT = vec<PtrT, Ns...>;
          RetT ret{};
          auto ptr = (PtrT)_base + (_idx / lane_width * _numChannels + _chn) * lane_width
                     + (_idx % lane_width);
          for (channel_counter_type d = 0; d != extent; ++d, ptr += lane_width) ret._data[d] = ptr;
          return ret;
        }
      }
      constexpr bool equal_to(iterator_impl it) const noexcept {
        return it._idx == _idx && it._chn == _chn;
      }
      constexpr void advance(difference_type offset) noexcept { _idx += offset; }
      constexpr difference_type distance_to(iterator_impl it) const noexcept {
        return it._idx - _idx;
      }

    protected:
      conditional_t<is_const_structure, const_pointer, pointer> _base{nullptr};
      size_type _idx{0};
      channel_counter_type _chn{0}, _numChannels{1};
    };

    template <typename ValT, auto... Ns>
    struct iterator_impl<ValT, value_seq<Ns...>,
                         enable_if_type<(sizeof(value_type) * lane_width % sizeof(ValT) == 0)
                                        && (sizeof(value_type) > sizeof(ValT))>>
        : IteratorInterface<
              iterator_impl<ValT, value_seq<Ns...>,
                            enable_if_type<(sizeof(value_type) * lane_width % sizeof(ValT) == 0)
                                           && (sizeof(value_type) > sizeof(ValT))>>> {
      static_assert(!std::is_reference_v<ValT>,
                    "the access type of the iterator should not be a reference.");
      static constexpr auto extent = (Ns * ...);
      static_assert(extent > 0, "the access extent of the iterator should be positive.");

      using iter_value_type = remove_cvref_t<ValT>;
      static constexpr bool is_const_structure = is_const_v<ValT>;
      static constexpr bool is_scalar_access = (extent == 1);

      static constexpr size_t num_segments
          = sizeof(value_type) * lane_width / sizeof(iter_value_type);

      constexpr iterator_impl(conditional_t<is_const_structure, const_pointer, pointer> base,
                              size_type idx, size_type segNo, channel_counter_type chn,
                              channel_counter_type nchns) noexcept
          : _base{base}, _idx{idx}, _segOffset{segNo * lane_width}, _chn{chn}, _numChannels{nchns} {
        if (segNo >= num_segments) throw std::runtime_error("not a valid segment index.");
      }

      constexpr decltype(auto) dereference() const {
        /// @note ref: https://en.cppreference.com/w/cpp/language/reinterpret_cast
        if constexpr (is_scalar_access) {
          return *((conditional_t<is_const_structure, const iter_value_type *,
                                  iter_value_type *>)(_base
                                                      + (_idx / lane_width * _numChannels + _chn)
                                                            * lane_width)
                   + _segOffset + _idx % lane_width);
        } else {
          using PtrT
              = conditional_t<is_const_structure, const iter_value_type *, iter_value_type *>;
          using RetT = vec<PtrT, Ns...>;
          RetT ret{};
          auto ptr = (PtrT)(_base + (_idx / lane_width * _numChannels + _chn) * lane_width)
                     + _segOffset + _idx % lane_width;
          for (channel_counter_type d = 0; d != extent; ++d, ptr += lane_width * num_segments)
            ret._data[d] = ptr;
          return ret;
        }
      }
      constexpr bool equal_to(iterator_impl it) const noexcept {
        return it._idx == _idx && it._chn == _chn;
      }
      constexpr void advance(difference_type offset) noexcept { _idx += offset; }
      constexpr difference_type distance_to(iterator_impl it) const noexcept {
        return it._idx - _idx;
      }

    protected:
      conditional_t<is_const_structure, const_pointer, pointer> _base{nullptr};
      size_type _idx{0}, _segOffset{0};
      channel_counter_type _chn{0}, _numChannels{1};
    };

    template <typename T = value_type, typename Dims = value_seq<1>> using iterator
        = LegacyIterator<iterator_impl<T, Dims>>;
    template <typename T = const value_type, typename Dims = value_seq<1>> using const_iterator
        = LegacyIterator<iterator_impl<std::add_const_t<T>, Dims>>;

    // for serialization
    constexpr auto begin() { return _buffer.begin(); }
    constexpr auto begin() const { return _buffer.begin(); }
    constexpr auto end() { return _buffer.end(); }
    constexpr auto end() const { return _buffer.end(); }
    // size-identical value iterator
    template <typename Dims = value_seq<1>, typename T = const value_type,
              enable_if_t<sizeof(T) == sizeof(value_type)> = 0>
    constexpr auto begin(channel_counter_type chn, Dims = {}, wrapt<T> = {}) noexcept {
      return make_iterator<iterator_impl<T, Dims>>(data(), static_cast<size_type>(0), chn,
                                                   numChannels());
    }
    template <typename Dims = value_seq<1>, typename T = const value_type,
              enable_if_t<sizeof(T) == sizeof(value_type)> = 0>
    constexpr auto end(channel_counter_type chn, Dims = {}, wrapt<T> = {}) noexcept {
      return make_iterator<iterator_impl<T, Dims>>(data(), size(), chn, numChannels());
    }
    template <typename Dims = value_seq<1>, typename T = const value_type,
              enable_if_t<sizeof(T) == sizeof(value_type)> = 0>
    constexpr auto begin(channel_counter_type chn, Dims = {}, wrapt<T> = {}) const noexcept {
      return make_iterator<iterator_impl<std::add_const_t<T>, Dims>>(
          data(), static_cast<size_type>(0), chn, numChannels());
    }
    template <typename Dims = value_seq<1>, typename T = const value_type,
              enable_if_t<sizeof(T) == sizeof(value_type)> = 0>
    constexpr auto end(channel_counter_type chn, Dims = {}, wrapt<T> = {}) const noexcept {
      return make_iterator<iterator_impl<std::add_const_t<T>, Dims>>(data(), size(), chn,
                                                                     numChannels());
    }
    template <typename Dims = value_seq<1>, typename T = value_type,
              enable_if_t<sizeof(T) == sizeof(value_type)> = 0>
    constexpr auto begin(const SmallString &prop, Dims = {}, wrapt<T> = {}) noexcept {
      return make_iterator<iterator_impl<T, Dims>>(data(), static_cast<size_type>(0),
                                                   getPropertyOffset(prop), numChannels());
    }
    template <typename Dims = value_seq<1>, typename T = value_type,
              enable_if_t<sizeof(T) == sizeof(value_type)> = 0>
    constexpr auto end(const SmallString &prop, Dims = {}, wrapt<T> = {}) noexcept {
      return make_iterator<iterator_impl<T, Dims>>(data(), size(), getPropertyOffset(prop),
                                                   numChannels());
    }
    template <typename Dims = value_seq<1>, typename T = const value_type,
              enable_if_t<sizeof(T) == sizeof(value_type)> = 0>
    constexpr auto begin(const SmallString &prop, Dims = {}, wrapt<T> = {}) const noexcept {
      return make_iterator<iterator_impl<std::add_const_t<T>, Dims>>(
          data(), static_cast<size_type>(0), getPropertyOffset(prop), numChannels());
    }
    template <typename Dims = value_seq<1>, typename T = const value_type,
              enable_if_t<sizeof(T) == sizeof(value_type)> = 0>
    constexpr auto end(const SmallString &prop, Dims = {}, wrapt<T> = {}) const noexcept {
      return make_iterator<iterator_impl<std::add_const_t<T>, Dims>>(
          data(), size(), getPropertyOffset(prop), numChannels());
    }
    // size-varying value iterator
    template <typename Dims = value_seq<1>, typename T = const value_type,
              enable_if_t<(sizeof(value_type) > sizeof(T))> = 0>
    constexpr auto begin(channel_counter_type chn, size_type segNo = 0, Dims = {},
                         wrapt<T> = {}) noexcept {
      return make_iterator<iterator_impl<T, Dims>>(data(), static_cast<size_type>(0), segNo, chn,
                                                   numChannels());
    }
    template <typename Dims = value_seq<1>, typename T = const value_type,
              enable_if_t<(sizeof(value_type) > sizeof(T))> = 0>
    constexpr auto end(channel_counter_type chn, size_type segNo = 0, Dims = {},
                       wrapt<T> = {}) noexcept {
      return make_iterator<iterator_impl<T, Dims>>(data(), size(), segNo, chn, numChannels());
    }
    template <typename Dims = value_seq<1>, typename T = const value_type,
              enable_if_t<(sizeof(value_type) > sizeof(T))> = 0>
    constexpr auto begin(channel_counter_type chn, size_type segNo = 0, Dims = {},
                         wrapt<T> = {}) const noexcept {
      return make_iterator<iterator_impl<std::add_const_t<T>, Dims>>(
          data(), static_cast<size_type>(0), segNo, chn, numChannels());
    }
    template <typename Dims = value_seq<1>, typename T = const value_type,
              enable_if_t<(sizeof(value_type) > sizeof(T))> = 0>
    constexpr auto end(channel_counter_type chn, size_type segNo = 0, Dims = {},
                       wrapt<T> = {}) const noexcept {
      return make_iterator<iterator_impl<std::add_const_t<T>, Dims>>(data(), size(), segNo, chn,
                                                                     numChannels());
    }
    template <typename Dims = value_seq<1>, typename T = value_type,
              enable_if_t<(sizeof(value_type) > sizeof(T))> = 0>
    constexpr auto begin(const SmallString &prop, size_type segNo, Dims = {},
                         wrapt<T> = {}) noexcept {
      return make_iterator<iterator_impl<T, Dims>>(data(), static_cast<size_type>(0), segNo,
                                                   getPropertyOffset(prop), numChannels());
    }
    template <typename Dims = value_seq<1>, typename T = value_type,
              enable_if_t<(sizeof(value_type) > sizeof(T))> = 0>
    constexpr auto end(const SmallString &prop, size_type segNo, Dims = {},
                       wrapt<T> = {}) noexcept {
      return make_iterator<iterator_impl<T, Dims>>(data(), size(), segNo, getPropertyOffset(prop),
                                                   numChannels());
    }
    template <typename Dims = value_seq<1>, typename T = const value_type,
              enable_if_t<(sizeof(value_type) > sizeof(T))> = 0>
    constexpr auto begin(const SmallString &prop, size_type segNo, Dims = {},
                         wrapt<T> = {}) const noexcept {
      return make_iterator<iterator_impl<std::add_const_t<T>, Dims>>(
          data(), static_cast<size_type>(0), segNo, getPropertyOffset(prop), numChannels());
    }
    template <typename Dims = value_seq<1>, typename T = const value_type,
              enable_if_t<(sizeof(value_type) > sizeof(T))> = 0>
    constexpr auto end(const SmallString &prop, size_type segNo, Dims = {},
                       wrapt<T> = {}) const noexcept {
      return make_iterator<iterator_impl<std::add_const_t<T>, Dims>>(
          data(), size(), segNo, getPropertyOffset(prop), numChannels());
    }

    /// capacity
    constexpr size_type size() const noexcept { return _size; }
    constexpr size_type capacity() const noexcept { return _buffer.capacity() / _numChannels; }
    constexpr channel_counter_type numChannels() const noexcept { return _numChannels; }
    constexpr size_type numTiles() const noexcept { return (size() + lane_width - 1) / lane_width; }
    constexpr size_type numReservedTiles() const noexcept {
      return (capacity() + lane_width - 1) / lane_width;
    }
    constexpr size_type tileBytes() const noexcept {
      return numChannels() * lane_width * sizeof(value_type);
    }
    constexpr bool empty() noexcept { return size() == 0; }

    /// element access
    constexpr reference operator[](
        const zs::tuple<channel_counter_type, size_type> index) noexcept {
      const auto [chn, idx] = index;
      return *(data() + (idx / lane_width * numChannels() + chn) * lane_width + idx % lane_width);
    }
    constexpr conditional_t<std::is_fundamental_v<value_type>, value_type, const_reference>
    operator[](const zs::tuple<channel_counter_type, size_type> index) const noexcept {
      const auto [chn, idx] = index;
      return *(data() + (idx / lane_width * numChannels() + chn) * lane_width + idx % lane_width);
    }
    /// tile offset
    constexpr const void *tileOffset(size_type tileNo) const noexcept {
      return static_cast<const void *>(data() + tileNo * numChannels() * lane_width);
    }
    constexpr void *tileOffset(size_type tileNo) noexcept {
      return static_cast<void *>(data() + tileNo * numChannels() * lane_width);
    }
    /// ctor, assignment operator
    TileVector(const TileVector &o)
        : _allocator{o._allocator},
          _buffer{o._buffer},
          _tags{o._tags},
          _tagNames{o._tagNames},
          _tagSizes{o._tagSizes},
          _tagOffsets{o._tagOffsets},
          _size{o.size()},
          _numChannels{o.numChannels()} {
      if (capacity() > 0)
        Resource::copy(MemoryEntity{memoryLocation(), (void *)data()},
                       MemoryEntity{o.memoryLocation(), (void *)o.data()},
                       sizeof(value_type) * o.numChannels() * o.capacity());
    }
    TileVector &operator=(const TileVector &o) {
      if (this == &o) return *this;
      TileVector tmp(o);
      swap(tmp);
      return *this;
    }
    TileVector clone(const allocator_type &allocator) const {
      TileVector ret{allocator, _tags, size()};
      Resource::copy(MemoryEntity{allocator.location, (void *)ret.data()},
                     MemoryEntity{memoryLocation(), (void *)data()},
                     sizeof(value_type) * numChannels() * (count_tiles(size()) * lane_width));
      return ret;
    }
    TileVector clone(const MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }
    /// assignment or destruction after move
    /// https://www.youtube.com/watch?v=ZG59Bqo7qX4
    /// explicit noexcept
    /// leave the source object in a valid (default constructed) state
    TileVector(TileVector &&o) noexcept {
      const TileVector defaultVector{};
      _buffer = zs::exchange(o._buffer, defaultVector._buffer);
      _allocator = zs::exchange(o._allocator, defaultVector._allocator);
      _tags = zs::exchange(o._tags, defaultVector._tags);
      _tagNames = zs::exchange(o._tagNames, defaultVector._tagNames);
      _tagSizes = zs::exchange(o._tagSizes, defaultVector._tagSizes);
      _tagOffsets = zs::exchange(o._tagOffsets, defaultVector._tagOffsets);
      _size = zs::exchange(o._size, defaultVector.size());
      _numChannels = zs::exchange(o._numChannels, defaultVector.numChannels());
    }
    /// make move-assignment safe for self-assignment
    TileVector &operator=(TileVector &&o) noexcept {
      if (this == &o) return *this;
      TileVector tmp(zs::move(o));
      swap(tmp);
      return *this;
    }
    void swap(TileVector &o) noexcept {
      std::swap(_buffer, o._buffer);
      std::swap(_allocator, o._allocator);
      std::swap(_tags, o._tags);
      std::swap(_tagNames, o._tagNames);
      std::swap(_tagSizes, o._tagSizes);
      std::swap(_tagOffsets, o._tagOffsets);
      std::swap(_size, o._size);
      std::swap(_numChannels, o._numChannels);
    }
    friend void swap(TileVector &a, TileVector &b) noexcept { a.swap(b); }

    void clear() { *this = TileVector{_allocator, _tags, 0}; }
    void resize(size_type newSize) {
      _size = newSize;
      newSize = count_tiles(newSize) * lane_width;
      _buffer.alignedResize(newSize * static_cast<size_type>(_numChannels),
                            lane_width * static_cast<size_type>(_numChannels));
    }
    void reset(int ch) {
      _buffer.reset(ch);
      // Resource::memset(MemoryEntity{memoryLocation(), (void *)data()}, ch,
      //                  numTiles() * tileBytes());
    }

    template <typename Policy>
    void append_channels(Policy &&, const std::vector<PropertyTag> &tags,
                         const source_location &loc = source_location::current());
    template <typename Policy> void reset(Policy &&policy, value_type val);
    template <typename Policy, typename MapRange, bool Scatter = true>
    void reorderTiles(Policy &&pol, MapRange &&mapR, wrapv<Scatter> = {});

    constexpr channel_counter_type numProperties() const noexcept { return _tags.size(); }

    bool hasProperty(const SmallString &str) const {
      for (auto &&tag : _tags)
        if (str == tag.name) return true;
      return false;
    }
    constexpr const SmallString *tagNameHandle() const noexcept { return _tagNames.data(); }
    constexpr const channel_counter_type *tagSizeHandle() const noexcept {
      return _tagSizes.data();
    }
    constexpr const channel_counter_type *tagOffsetHandle() const noexcept {
      return _tagOffsets.data();
    }
    [[deprecated]] channel_counter_type getChannelSize(const SmallString &str) const {
      for (auto &&tag : _tags)
        if (str == tag.name) return tag.numChannels;
      return 0;
    }
    channel_counter_type getPropertySize(const SmallString &str) const {
      for (auto &&tag : _tags)
        if (str == tag.name) return tag.numChannels;
      return 0;
    }
    [[deprecated]] channel_counter_type getChannelOffset(const SmallString &str) const {
      channel_counter_type offset = 0;
      for (auto &&tag : _tags) {
        if (str == tag.name) return offset;
        offset += tag.numChannels;
      }
      return 0;
    }
    channel_counter_type getPropertyOffset(const SmallString &str) const {
      channel_counter_type offset = 0;
      for (auto &&tag : _tags) {
        if (str == tag.name) return offset;
        offset += tag.numChannels;
      }
      return -1;
    }
    constexpr PropertyTag getPropertyTag(size_t i = 0) const { return _tags[i]; }
    constexpr const auto &getPropertyTags() const { return _tags; }

    auto &refBuffer() { return _buffer; }
    auto &refTags() { return _tags; }
    auto &refTagNames() { return _tagNames; }
    auto &refTagSizes() { return _tagSizes; }
    auto &refTagOffsets() { return _tagOffsets; }
    auto &refSize() { return _size; }
    auto &refNumChannels() { return _numChannels; }
    size_type bufferSize() const { return _buffer.size(); }
    void resizeBuffer(size_type bufferSize) {
      _buffer.resizeBuffer(bufferSize, lane_width * static_cast<size_type>(_numChannels));
    }

  protected:
    allocator_type _allocator{};
    Vector<value_type, allocator_type> _buffer{};
    std::vector<PropertyTag> _tags{};  // on host
    /// for proxy use
    Vector<SmallString, allocator_type> _tagNames{};
    Vector<channel_counter_type, allocator_type> _tagSizes{};
    Vector<channel_counter_type, allocator_type> _tagOffsets{};
    size_type _size{0};                    // element size
    channel_counter_type _numChannels{1};  // this must be serialized ahead
  };

#define ZS_FWD_DECL_TILEVECTOR_INSTANTIATIONS(LENGTH)                         \
  ZPC_FWD_DECL_TEMPLATE_STRUCT TileVector<u32, LENGTH, ZSPmrAllocator<>>;     \
  ZPC_FWD_DECL_TEMPLATE_STRUCT TileVector<u64, LENGTH, ZSPmrAllocator<>>;     \
  ZPC_FWD_DECL_TEMPLATE_STRUCT TileVector<i32, LENGTH, ZSPmrAllocator<>>;     \
  ZPC_FWD_DECL_TEMPLATE_STRUCT TileVector<i64, LENGTH, ZSPmrAllocator<>>;     \
  ZPC_FWD_DECL_TEMPLATE_STRUCT TileVector<f32, LENGTH, ZSPmrAllocator<>>;     \
  ZPC_FWD_DECL_TEMPLATE_STRUCT TileVector<f64, LENGTH, ZSPmrAllocator<>>;     \
  ZPC_FWD_DECL_TEMPLATE_STRUCT TileVector<u32, LENGTH, ZSPmrAllocator<true>>; \
  ZPC_FWD_DECL_TEMPLATE_STRUCT TileVector<u64, LENGTH, ZSPmrAllocator<true>>; \
  ZPC_FWD_DECL_TEMPLATE_STRUCT TileVector<i32, LENGTH, ZSPmrAllocator<true>>; \
  ZPC_FWD_DECL_TEMPLATE_STRUCT TileVector<i64, LENGTH, ZSPmrAllocator<true>>; \
  ZPC_FWD_DECL_TEMPLATE_STRUCT TileVector<f32, LENGTH, ZSPmrAllocator<true>>; \
  ZPC_FWD_DECL_TEMPLATE_STRUCT TileVector<f64, LENGTH, ZSPmrAllocator<true>>;

  /// 8, 32, 64, 512
  ZS_FWD_DECL_TILEVECTOR_INSTANTIATIONS(8)
  ZS_FWD_DECL_TILEVECTOR_INSTANTIATIONS(32)
  ZS_FWD_DECL_TILEVECTOR_INSTANTIATIONS(64)
  ZS_FWD_DECL_TILEVECTOR_INSTANTIATIONS(512)

  template <typename TileVectorView> struct TileVectorCopy {
    using size_type = typename TileVectorView::size_type;
    using channel_counter_type = typename TileVectorView::channel_counter_type;
    TileVectorCopy(TileVectorView src, TileVectorView dst) : src{src}, dst{dst} {}
    constexpr void operator()(size_type i) {
      const auto nchns = src.numChannels();
      channel_counter_type chn = 0;
      for (; chn != nchns; ++chn) dst(chn, i) = src(chn, i);
      /// zero-initialize newly appended channels
      // note: do not rely on this convention!
      const auto totalchns = dst.numChannels();
      for (; chn != totalchns; ++chn) dst(chn, i) = 0;
    }
    TileVectorView src, dst;
  };
  template <typename T, size_t Length, typename Allocator> template <typename Policy>
  void TileVector<T, Length, Allocator>::append_channels(Policy &&policy,
                                                         const std::vector<PropertyTag> &appendTags,
                                                         const source_location &loc) {
    constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
    const auto s = size();
    auto tags = getPropertyTags();
    bool modified = false;
    for (const auto &tag : appendTags) {
      size_t i = 0;
      for (; i != tags.size(); ++i)
        if (tags[i].name == tag.name) break;
      if (i == tags.size()) {
        tags.push_back(tag);
        modified = true;
      } else if (tags[i].numChannels != tag.numChannels)
        throw std::runtime_error(
            fmt::format("append_channels: property[{}] currently has [{}] channels, cannot change "
                        "it to [{}] channels.",
                        tag.name.asChars(), tags[i].numChannels, tag.numChannels));
    }
    if (!modified) return;
    TileVector<T, Length, Allocator> tmp{get_allocator(), tags, s};
    policy(range(s), TileVectorCopy{proxy<space>(*this), proxy<space>(tmp)}, loc);
    *this = zs::move(tmp);
  }
  template <typename TileVectorView> struct TileVectorReset {
    using size_type = typename TileVectorView::size_type;
    using value_type = typename TileVectorView::value_type;
    using channel_counter_type = typename TileVectorView::channel_counter_type;
    TileVectorReset(TileVectorView tv, value_type val) : tv{tv}, v{val} {}
    constexpr void operator()(size_type i) {
      const auto nchns = tv.numChannels();
      for (channel_counter_type chn = 0; chn != nchns; ++chn) tv(chn, i) = v;
    }
    TileVectorView tv;
    value_type v;
  };
  template <typename T, size_t Length, typename Allocator> template <typename Policy>
  void TileVector<T, Length, Allocator>::reset(Policy &&policy, value_type val) {
    constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
    policy(range(size()), TileVectorReset{proxy<space>(*this), val});
  }
  template <typename TileVectorView, typename MapIter, bool Scatter> struct TileVectorTileReorder {
    using size_type = typename TileVectorView::size_type;
    using value_type = typename TileVectorView::value_type;
    static constexpr auto lane_width = TileVectorView::lane_width;
    using channel_counter_type = typename TileVectorView::channel_counter_type;
    TileVectorTileReorder(TileVectorView tiles, TileVectorView orderedTiles, MapIter map)
        : tiles{tiles}, orderedTiles{orderedTiles}, map{map} {}
    constexpr void operator()(size_type i) {
      const auto nchns = tiles.numChannels();
      auto offset = i % lane_width;
      i /= lane_width;       // tile-i
      size_type j = map[i];  // tile-j
      // reorder tiles
      if constexpr (Scatter) {  // scatter
        auto srcTile = tiles.tile(i);
        auto dstTile = orderedTiles.tile(j);
        for (channel_counter_type d = 0; d != nchns; ++d) dstTile(d, offset) = srcTile(d, offset);
      } else {  // gather
        auto srcTile = tiles.tile(j);
        auto dstTile = orderedTiles.tile(i);
        for (channel_counter_type d = 0; d != nchns; ++d) dstTile(d, offset) = srcTile(d, offset);
      }
    }
    TileVectorView tiles, orderedTiles;
    MapIter map;
  };
  template <typename T, size_t Length, typename Allocator>
  template <typename Policy, typename MapRange, bool Scatter>
  void TileVector<T, Length, Allocator>::reorderTiles(Policy &&pol, MapRange &&mapR,
                                                      wrapv<Scatter>) {
    constexpr execspace_e space = RM_REF_T(pol)::exec_tag::value;
    using Ti = RM_CVREF_T(*zs::begin(mapR));
    static_assert(is_integral_v<Ti>,
                  "index mapping range\'s dereferenced type is not an integral.");

    const size_type sz = numTiles();
    if (range_size(mapR) != sz) throw std::runtime_error("index mapping range size mismatch");
    if (!valid_memspace_for_execution(pol, get_allocator()))
      throw std::runtime_error("current memory location not compatible with the execution policy");

    TileVector orderedTiles{get_allocator(), getPropertyTags(), size()};
    {
      auto tiles = view<space>(*this);
      auto oTiles = view<space>(orderedTiles);
      auto mapIter = zs::begin(mapR);
      pol(range(sz * lane_width),
          TileVectorTileReorder<RM_CVREF_T(tiles), RM_CVREF_T(mapIter), Scatter>{tiles, oTiles,
                                                                                 mapIter});
    }
    *this = zs::move(orderedTiles);
  }

  template <execspace_e Space, typename TileVectorT, bool WithinTile, bool Base = false,
            typename = void>
  struct TileVectorUnnamedView {
    static constexpr auto space = Space;
    static constexpr bool is_const_structure = is_const_v<TileVectorT>;
    using tile_vector_type = remove_const_t<TileVectorT>;
    using const_tile_vector_type = std::add_const_t<tile_vector_type>;
    using pointer = typename tile_vector_type::pointer;
    using const_pointer = typename tile_vector_type::const_pointer;
    using value_type = typename tile_vector_type::value_type;
    using reference = typename tile_vector_type::reference;
    using const_reference = typename tile_vector_type::const_reference;
    using size_type = typename tile_vector_type::size_type;
    using difference_type = typename tile_vector_type::difference_type;
    using channel_counter_type = typename tile_vector_type::channel_counter_type;
    using whole_view_type = TileVectorUnnamedView<Space, TileVectorT, false, Base>;
    using tile_view_type = TileVectorUnnamedView<Space, TileVectorT, true, Base>;
    static constexpr auto lane_width = tile_vector_type::lane_width;

    TileVectorUnnamedView() noexcept = default;
    explicit constexpr TileVectorUnnamedView(TileVectorT &tilevector)
        : _vector{tilevector.data()}, _dims{tilevector.size(), tilevector.numChannels()} {}
    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    explicit constexpr TileVectorUnnamedView(pointer base, const size_type s,
                                             const channel_counter_type nchns)
        : _vector{base}, _dims{s, nchns} {}
    template <bool V = is_const_structure, enable_if_t<V> = 0>
    explicit constexpr TileVectorUnnamedView(const_pointer base, const size_type s,
                                             const channel_counter_type nchns)
        : _vector{base}, _dims{s, nchns} {}

    template <bool Pred = (!WithinTile), enable_if_t<Pred> = 0>
    constexpr size_type numTiles() const noexcept {
      /// @note size can be as large as limits<size_type>::max()!
      /// @note therefore might be inaccurate when Base==true
      const auto s = _dims.size();
      return s / lane_width + (s % lane_width > 0 ? 1 : 0);
    }
    template <bool V = is_const_structure, typename TT = value_type,
              enable_if_all<!V, sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr std::add_lvalue_reference_t<TT> operator()(const channel_counter_type chn,
                                                         const size_type i,
                                                         wrapt<TT> = {}) noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if ((TT *)_vector == nullptr) {
        /// @note TBD : insert type reflection info here
        printf("tilevector [%s] operator() reinterpret_cast failed!\n", _nameTag.asChars());
        return *((TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
      }
      if (chn >= _dims._numChannels) {
        printf("tilevector [%s] ofb! accessing chn [%d] out of [0, %d)\n", _nameTag.asChars(),
               (int)chn, (int)_dims._numChannels);
        return *((TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
      }
#endif
      if constexpr (WithinTile) {
#if ZS_ENABLE_OFB_ACCESS_CHECK
        if (i >= lane_width) {
          printf("tilevector [%s] ofb! in-tile accessing ele [%lld] out of [0, %lld)\n",
                 _nameTag.asChars(), (long long)i, (long long)lane_width);
          return *((TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
        }
#endif

        return *((TT *)_vector + (chn * lane_width + i));
      } else {
#if ZS_ENABLE_OFB_ACCESS_CHECK
        if (i >= _dims.size()) {
          printf("tilevector [%s] ofb! global accessing ele [%lld] out of [0, %lld)\n",
                 _nameTag.asChars(), (long long)i, (long long)_dims.size());
          return *((TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
        }
#endif
        return *((TT *)_vector
                 + ((i / lane_width * _dims._numChannels + chn) * lane_width + i % lane_width));
      }
    }
    template <typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr TT operator()(const channel_counter_type chn, const size_type i,
                            wrapt<TT> = {}) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if ((const TT *)_vector == nullptr) {
        /// @note TBD : insert type reflection info here
        printf("tilevector [%s] operator() reinterpret_cast failed!\n", _nameTag.asChars());
        return *((const TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
      }
      if (chn >= _dims._numChannels) {
        printf("tilevector [%s] ofb! accessing chn [%d] out of [0, %d)\n", _nameTag.asChars(),
               (int)chn, (int)_dims._numChannels);
        return *((const TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
      }
#endif
      if constexpr (WithinTile) {
#if ZS_ENABLE_OFB_ACCESS_CHECK
        if (i >= lane_width) {
          printf("tilevector [%s] ofb! in-tile accessing ele [%lld] out of [0, %lld)\n",
                 _nameTag.asChars(), (long long)i, (long long)lane_width);
          return *((const TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
        }
#endif
        return *((const TT *)_vector + (chn * lane_width + i));
      } else {
#if ZS_ENABLE_OFB_ACCESS_CHECK
        if (i >= _dims.size()) {
          printf("tilevector [%s] ofb! global accessing ele [%lld] out of [0, %lld)\n",
                 _nameTag.asChars(), (long long)i, (long long)_dims.size());
          return *((const TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
        }
#endif
        return *((const TT *)_vector
                 + ((i / lane_width * _dims._numChannels + chn) * lane_width + i % lane_width));
      }
    }

    template <typename TT = value_type, bool V = is_const_structure, bool InTile = WithinTile,
              enable_if_all<!V, !InTile, sizeof(TT) == sizeof(value_type),
                            is_same_v<TT, remove_cvref_t<TT>>, (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr std::add_lvalue_reference_t<TT> operator()(const channel_counter_type chn,
                                                         const size_type tileNo,
                                                         const size_type localNo,
                                                         wrapt<TT> = {}) noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if ((TT *)_vector == nullptr) {
        /// @note TBD : insert type reflection info here
        printf("tilevector [%s] operator()[tile] reinterpret_cast failed!\n", _nameTag.asChars());
        return *((TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
      }
      if (chn >= _dims._numChannels) {
        printf("tilevector [%s] ofb! accessing chn [%d] out of [0, %d)\n", _nameTag.asChars(),
               (int)chn, (int)_dims._numChannels);
        return *((TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
      }
      if (localNo >= lane_width) {
        printf("tilevector [%s] ofb! local accessing ele [%lld] out of [0, %lld)\n",
               _nameTag.asChars(), (long long)tileNo, (long long)lane_width);
        return *((TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
      }
#endif
      return *((TT *)_vector + ((tileNo * _dims._numChannels + chn) * lane_width + localNo));
    }
    template <typename TT = value_type, bool InTile = WithinTile,
              enable_if_all<!InTile, sizeof(TT) == sizeof(value_type),
                            is_same_v<TT, remove_cvref_t<TT>>, (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr TT operator()(const channel_counter_type chn, const size_type tileNo,
                            const size_type localNo, wrapt<TT> = {}) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if ((const TT *)_vector == nullptr) {
        /// @note TBD : insert type reflection info here
        printf("tilevector [%s] operator()[tile] reinterpret_cast failed!\n", _nameTag.asChars());
        return *((const TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
      }
      if (chn >= _dims._numChannels) {
        printf("tilevector [%s] ofb! accessing chn [%d] out of [0, %d)\n", _nameTag.asChars(),
               (int)chn, (int)_dims._numChannels);
        return *((const TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
      }
      if (localNo >= lane_width) {
        printf("tilevector [%s] ofb! local accessing ele [%lld] out of [0, %lld)\n",
               _nameTag.asChars(), (long long)tileNo, (long long)lane_width);
        return *((const TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
      }
#endif
      return *((const TT *)_vector + ((tileNo * _dims._numChannels + chn) * lane_width + localNo));
    }

    template <bool V = is_const_structure, bool InTile = WithinTile, enable_if_all<!V, !InTile> = 0>
    constexpr auto tile(const size_type tileid) noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (long long nt = numTiles(); tileid >= nt) {
        printf("tilevector [%s] ofb! global accessing tile [%lld] out of [0, %lld)\n",
               _nameTag.asChars(), (long long)tileid, nt);
        return TileVectorUnnamedView<Space, tile_vector_type, true, Base>{
            (value_type *)0, lane_width, _dims._numChannels};
      }
#endif
      return TileVectorUnnamedView<Space, tile_vector_type, true, Base>{
          _vector + tileid * lane_width * _dims._numChannels, lane_width, _dims._numChannels};
    }
    template <bool InTile = WithinTile, enable_if_t<!InTile> = 0>
    constexpr auto tile(const size_type tileid) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (long long nt = numTiles(); tileid >= nt) {
        printf("tilevector [%s] ofb! global accessing tile [%lld] out of [0, %lld)\n",
               _nameTag.asChars(), (long long)tileid, nt);
        return TileVectorUnnamedView<Space, const_tile_vector_type, true, Base>{
            (const value_type *)0, lane_width, _dims._numChannels};
      }
#endif
      return TileVectorUnnamedView<Space, const_tile_vector_type, true, Base>{
          _vector + tileid * lane_width * _dims._numChannels, lane_width, _dims._numChannels};
    }

    // use dim_c<Ns...> for the first parameter
    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto pack(value_seq<Ns...>, channel_counter_type chn, const size_type i,
                        wrapt<TT> = {}) const noexcept {
      using RetT = vec<TT, Ns...>;
      RetT ret{};
#if ZS_ENABLE_OFB_ACCESS_CHECK
      /// @brief check reinterpret_cast result validity (bit_cast should be more robust)
      if ((const TT *)_vector == nullptr) {
        /// @note TBD : insert type reflection info here
        printf("tilevector [%s] packing reinterpret_cast failed!\n", _nameTag.asChars());
        return RetT::constant(
            *(const TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
      }
      /// @brief check channel access overflow
      if (chn + RetT::extent > _dims._numChannels) {
        printf("tilevector [%s] ofb! accessing chn [%d, %d) out of [0, %d)\n", _nameTag.asChars(),
               (int)chn, (int)(chn + RetT::extent), (int)_dims._numChannels);
        return RetT::constant(
            *(const TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
      }
#endif
      const TT *ptr = nullptr;
      if constexpr (WithinTile) {
        ptr = (const TT *)_vector + (chn * lane_width + i);
#if ZS_ENABLE_OFB_ACCESS_CHECK
        if (i >= lane_width) {
          printf("tilevector [%s] ofb! in-tile accessing ele [%lld] out of [0, %lld)\n",
                 _nameTag.asChars(), (long long)i, (long long)lane_width);
          return RetT::constant(
              *(const TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
        }
#endif
      } else {
        ptr = (const TT *)_vector
              + ((i / lane_width * _dims._numChannels + chn) * lane_width + (i % lane_width));
#if ZS_ENABLE_OFB_ACCESS_CHECK
        /// @brief check vector size overflow
        if (i >= _dims.size()) {
          printf("tilevector [%s] ofb! global accessing ele [%lld] out of [0, %lld)\n",
                 _nameTag.asChars(), (long long)i, (long long)_dims.size());
          return RetT::constant(
              *(const TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) + 1));
        }
#endif
      }
      for (channel_counter_type d = 0; d != RetT::extent; ++d, ptr += lane_width) ret.val(d) = *ptr;
      return ret;
    }
    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto pack(channel_counter_type chn, const size_type i,
                        wrapt<TT> = {}) const noexcept {
      return pack(dim_c<Ns...>, chn, i, wrapt<TT>{});
    }

    template <channel_counter_type N, typename VT = value_type>
    constexpr auto array(channel_counter_type chn, const size_type i) const noexcept {
      using RetT = std::array<VT, (size_t)N>;
      RetT ret{};
      auto ptr
          = _vector + ((i / lane_width * _dims._numChannels + chn) * lane_width + (i % lane_width));
      for (channel_counter_type d = 0; d != N; ++d, ptr += lane_width) ret[d] = *ptr;
      return ret;
    }
    /// tuple
    template <size_t... Is, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto tuple_impl(const channel_counter_type chnOffset, const size_type i,
                              index_sequence<Is...>, wrapt<TT>) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      constexpr channel_counter_type d = sizeof...(Is);
      if ((TT *)_vector == nullptr) {
        /// @note TBD : insert type reflection info here
        printf("tilevector [%s] tieing reinterpret_cast failed!\n", _nameTag.asChars());
        return zs::tie(
            *(TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) * (Is + 1) + 1)...);
      }
      if (chnOffset + d > _dims._numChannels) {
        printf("tilevector [%s] ofb! tieing chn [%d, %d) out of [0, %d)\n", _nameTag.asChars(),
               (int)chnOffset, (int)(chnOffset + d), (int)_dims._numChannels);
        return zs::tie(
            *(TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) * (Is + 1) + 1)...);
      }
      if constexpr (WithinTile) {
        if (i >= lane_width) {
          printf("tilevector [%s] ofb! in-tile accessing ele [%lld] out of [0, %lld)\n",
                 _nameTag.asChars(), (long long)i, (long long)lane_width);
          return zs::tie(
              *(TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) * (Is + 1) + 1)...);
        }
      } else {
        if (i >= _dims.size()) {
          printf("tilevector [%s] ofb! global tieing ele [%lld] out of [0, %lld)\n",
                 _nameTag.asChars(), (long long)i, (long long)_dims.size());
          return zs::tie(
              *(TT *)(detail::deduce_numeric_max<std::uintptr_t>() - sizeof(TT) * (Is + 1) + 1)...);
        }
      }
#endif
      if constexpr (WithinTile)
        return zs::tie(*((TT *)_vector
                         + ((size_type)chnOffset + (size_type)Is) * (size_type)lane_width + i)...);
      else {
        size_type a{}, b{};
        a = i / lane_width * _dims._numChannels;
        b = i % lane_width;
        return zs::tie(*((TT *)_vector
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
    template <auto d, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto tuple(const channel_counter_type chn, const size_type i,
                         wrapt<TT> = {}) const noexcept {
      return tuple(dim_c<d>, chn, i, wrapt<TT>{});
    }

    /// mount
    template <auto... Ns, typename TT, size_t... Is,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type)), (sizeof...(Is) == (Ns * ...))>
              = 0>
    constexpr auto mount_impl(value_seq<Ns...>, const channel_counter_type chnOffset,
                              const size_type i, wrapt<TT>, index_sequence<Is...>) const noexcept {
      // using R = zs::vec<const value_type *, (int)Ns...>;
      using R = zs::vec<TT *, (int)Ns...>;
      if constexpr (WithinTile)
        return R{*((TT *)_vector + ((size_type)chnOffset + (size_type)Is) * (size_type)lane_width
                   + i)...};
      else {
        size_type a{}, b{};
        a = i / lane_width * _dims._numChannels;
        b = i % lane_width;
        return R{*((TT *)_vector
                   + (a + ((size_type)chnOffset + (size_type)Is)) * (size_type)lane_width + b)...};
      }
    }
    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto mount(value_seq<Ns...>, const channel_counter_type chnOffset, const size_type i,
                         wrapt<TT> = {}) const noexcept {
      return mount_impl(value_seq<Ns...>{}, chnOffset, i, wrapt<TT>{},
                        make_index_sequence<(Ns * ...)>{});
    }

    constexpr size_type size() const noexcept { return _dims.size(); }
    constexpr channel_counter_type numChannels() const noexcept { return _dims._numChannels; }

  protected:
    /// @note explicit specialization within nested class not allowed, thus add 'Dummy'
    template <bool WithSize = false, typename Dummy = void> struct Dims {
      constexpr Dims() noexcept = default;
      ~Dims() noexcept = default;
      constexpr Dims(size_type s, channel_counter_type c) noexcept : _numChannels{c} {}
      constexpr size_type size() const noexcept {
        if constexpr (WithinTile)
          return lane_width;
        else
          return detail::deduce_numeric_max<size_type>();
      }
      channel_counter_type _numChannels{0};
    };
    template <typename Dummy> struct Dims<true, Dummy> {
      constexpr Dims() noexcept = default;
      ~Dims() noexcept = default;
      constexpr Dims(size_type s, channel_counter_type c) noexcept
          : _vectorSize{s}, _numChannels{c} {}
      constexpr size_type size() const noexcept { return _vectorSize; }
      size_type _vectorSize{0};
      channel_counter_type _numChannels{0};
    };

  public:
    conditional_t<is_const_structure, const_pointer, pointer> _vector{nullptr};

    Dims<(!Base && !WithinTile), void> _dims;
    // size_type _vectorSize{0};
    // channel_counter_type _numChannels{0};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    SmallString _nameTag{};
#endif
  };

  template <execspace_e ExecSpace, typename T, size_t Length, typename Allocator,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(const TileVector<T, Length, Allocator> &vec, wrapv<Base> = {}) {
    return TileVectorUnnamedView<ExecSpace, const TileVector<T, Length, Allocator>, false, Base>{
        vec};
  }
  template <execspace_e ExecSpace, typename T, size_t Length, typename Allocator,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(TileVector<T, Length, Allocator> &vec, wrapv<Base> = {}) {
    return TileVectorUnnamedView<ExecSpace, TileVector<T, Length, Allocator>, false, Base>{vec};
  }

  template <execspace_e ExecSpace, typename T, size_t Length, typename Allocator,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(const TileVector<T, Length, Allocator> &vec, wrapv<Base>,
                      const SmallString &tagName) {
    auto ret
        = TileVectorUnnamedView<ExecSpace, const TileVector<T, Length, Allocator>, false, Base>{
            vec};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ret._nameTag = tagName;
#endif
    return ret;
  }
  template <execspace_e ExecSpace, typename T, size_t Length, typename Allocator,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(TileVector<T, Length, Allocator> &vec, wrapv<Base>,
                      const SmallString &tagName) {
    auto ret = TileVectorUnnamedView<ExecSpace, TileVector<T, Length, Allocator>, false, Base>{vec};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ret._nameTag = tagName;
#endif
    return ret;
  }

  template <execspace_e space, typename T, size_t Length, typename Allocator>
  decltype(auto) proxy(const TileVector<T, Length, Allocator> &vec) {
    return view<space>(vec, false_c);
  }
  template <execspace_e space, typename T, size_t Length, typename Allocator>
  decltype(auto) proxy(TileVector<T, Length, Allocator> &vec) {
    return view<space>(vec, false_c);
  }

  template <execspace_e space, typename T, size_t Length, typename Allocator>
  decltype(auto) proxy(const TileVector<T, Length, Allocator> &vec, const SmallString &tagName) {
    return view<space>(vec, false_c, tagName);
  }
  template <execspace_e space, typename T, size_t Length, typename Allocator>
  decltype(auto) proxy(TileVector<T, Length, Allocator> &vec, const SmallString &tagName) {
    return view<space>(vec, false_c, tagName);
  }

  template <execspace_e Space, typename TileVectorT, bool WithinTile, bool Base = false,
            typename = void>
  struct TileVectorView : TileVectorUnnamedView<Space, TileVectorT, WithinTile, Base> {
    using base_t = TileVectorUnnamedView<Space, TileVectorT, WithinTile, Base>;
#if ZS_ENABLE_OFB_ACCESS_CHECK
    using base_t::_nameTag;
#endif

    static constexpr auto space = base_t::space;
    static constexpr bool is_const_structure = base_t::is_const_structure;
    using tile_vector_type = typename base_t::tile_vector_type;
    using const_tile_vector_type = typename base_t::const_tile_vector_type;
    using pointer = typename base_t::pointer;
    using const_pointer = typename base_t::const_pointer;
    using value_type = typename base_t::value_type;
    using reference = typename base_t::reference;
    using const_reference = typename base_t::const_reference;
    using size_type = typename base_t::size_type;
    using difference_type = typename base_t::difference_type;
    using channel_counter_type = typename base_t::channel_counter_type;
    using base_t::_dims;
    using whole_view_type = TileVectorView<Space, TileVectorT, false, Base>;
    using tile_view_type = TileVectorView<Space, TileVectorT, true, Base>;
    static constexpr auto lane_width = base_t::lane_width;

    TileVectorView() noexcept = default;
    explicit constexpr TileVectorView(const std::vector<SmallString> &tagNames,
                                      TileVectorT &tilevector)
        : base_t{tilevector},
          _tagNames{tilevector.tagNameHandle()},
          _tagOffsets{tilevector.tagOffsetHandle()},
          _tagSizes{tilevector.tagSizeHandle()},
          _N{static_cast<channel_counter_type>(tilevector.numProperties())} {}
    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    explicit constexpr TileVectorView(pointer base, const size_type s,
                                      const channel_counter_type nchns, const SmallString *tagNames,
                                      const channel_counter_type *tagOffsets,
                                      const channel_counter_type *tagSizes,
                                      const channel_counter_type N)
        : base_t{base, s, nchns},
          _tagNames{tagNames},
          _tagOffsets{tagOffsets},
          _tagSizes{tagSizes},
          _N{N} {}
    template <bool V = is_const_structure, enable_if_t<V> = 0>
    explicit constexpr TileVectorView(const_pointer base, const size_type s,
                                      const channel_counter_type nchns, const SmallString *tagNames,
                                      const channel_counter_type *tagOffsets,
                                      const channel_counter_type *tagSizes,
                                      const channel_counter_type N)
        : base_t{base, s, nchns},
          _tagNames{tagNames},
          _tagOffsets{tagOffsets},
          _tagSizes{tagSizes},
          _N{N} {}

    constexpr auto getPropertyNames() const noexcept { return _tagNames; }
    constexpr auto getPropertyOffsets() const noexcept { return _tagOffsets; }
    constexpr auto getPropertySizes() const noexcept { return _tagSizes; }
    constexpr auto numProperties() const noexcept { return _N; }
    constexpr auto propertyIndex(const SmallString &propName) const noexcept {
      channel_counter_type i = 0;
      for (; i != _N; ++i)
        if (_tagNames[i] == propName) break;
      return i;
    }
    constexpr auto propertySize(const SmallString &propName) const noexcept {
      return getPropertySizes()[propertyIndex(propName)];
    }
    constexpr auto propertyOffset(const SmallString &propName) const noexcept {
      return getPropertyOffsets()[propertyIndex(propName)];
    }
    constexpr bool hasProperty(const SmallString &propName) const noexcept {
      return propertyIndex(propName) != _N;
    }

    using base_t::operator();
    using base_t::mount;
    using base_t::numTiles;
    using base_t::pack;
    using base_t::tuple;
    ///
    /// have to make sure that char type (might be channel_counter_type) not fit into this overload
    ///
    template <bool V = is_const_structure, typename TT = value_type,
              enable_if_all<!V, sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr std::add_lvalue_reference_t<TT> operator()(const SmallString &propName,
                                                         const channel_counter_type chn,
                                                         const size_type i,
                                                         wrapt<TT> = {}) noexcept {
      auto propNo = propertyIndex(propName);
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (propNo == _N) {
        printf(
            "tilevector [%s] ofb! (operator()) accessing prop [%s] which is not among %d props (%d "
            "chns, %lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_dims._numChannels,
            (long long int)_dims.size());
        return static_cast<base_t &>(*this)(_dims._numChannels, i, wrapt<TT>{});
      }
#endif
      return static_cast<base_t &>(*this)(_tagOffsets[propNo] + chn, i, wrapt<TT>{});
    }
    template <typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr TT operator()(const SmallString &propName, const channel_counter_type chn,
                            const size_type i, wrapt<TT> = {}) const noexcept {
      auto propNo = propertyIndex(propName);
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (propNo == _N) {
        printf(
            "tilevector [%s] ofb! (operator()) const accessing prop [%s] which is not among %d "
            "props (%d chns, %lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_dims._numChannels,
            (long long int)_dims.size());
        return static_cast<const base_t &>(*this)(_dims._numChannels, i, wrapt<TT>{});
      }
#endif
      return static_cast<const base_t &>(*this)(_tagOffsets[propNo] + chn, i, wrapt<TT>{});
    }
    template <bool V = is_const_structure, typename TT = value_type,
              enable_if_all<!V, sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr std::add_lvalue_reference_t<TT> operator()(const SmallString &propName,
                                                         const size_type i,
                                                         wrapt<TT> = {}) noexcept {
      auto propNo = propertyIndex(propName);
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (propNo == _N) {
        printf(
            "tilevector [%s] ofb! (operator()) accessing prop [%s] which is not among %d props (%d "
            "chns, %lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_dims._numChannels,
            (long long int)_dims.size());
        return static_cast<base_t &>(*this)(_dims._numChannels, i, wrapt<TT>{});
      }
#endif
      return static_cast<base_t &>(*this)(_tagOffsets[propNo], i, wrapt<TT>{});
    }
    template <typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr TT operator()(const SmallString &propName, const size_type i,
                            wrapt<TT> = {}) const noexcept {
      auto propNo = propertyIndex(propName);
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (propNo == _N) {
        printf(
            "tilevector [%s] ofb! (operator()) const accessing prop [%s] which is not among %d "
            "props (%d chns, %lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_dims._numChannels,
            (long long int)_dims.size());
        return static_cast<const base_t &>(*this)(_dims._numChannels, i, wrapt<TT>{});
      }
#endif
      return static_cast<const base_t &>(*this)(_tagOffsets[propNo], i, wrapt<TT>{});
    }
    template <bool V = is_const_structure, bool InTile = WithinTile, enable_if_all<!V, !InTile> = 0>
    constexpr auto tile(const size_type tileid) noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (long long nt = numTiles(); tileid >= nt) {
        printf("tilevector [%s] ofb! global accessing tile %lld out of %lld blocks\n",
               _nameTag.asChars(), (long long)tileid, nt);
      }
#endif
      return TileVectorView<Space, tile_vector_type, true, Base>{
          this->_vector + tileid * lane_width * _dims._numChannels,
          lane_width,
          _dims._numChannels,
          _tagNames,
          _tagOffsets,
          _tagSizes,
          _N};
    }
    template <bool InTile = WithinTile, enable_if_t<!InTile> = 0>
    constexpr auto tile(const size_type tileid) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (long long nt = numTiles(); tileid >= nt) {
        printf("tilevector [%s] ofb! const global accessing tile %lld out of %lld blocks\n",
               _nameTag.asChars(), (long long)tileid, nt);
      }
#endif
      return TileVectorView<Space, const_tile_vector_type, true, Base>{
          this->_vector + tileid * lane_width * _dims._numChannels,
          lane_width,
          _dims._numChannels,
          _tagNames,
          _tagOffsets,
          _tagSizes,
          _N};
    }

    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto pack(value_seq<Ns...>, const SmallString &propName, const size_type i,
                        wrapt<TT> = {}) const noexcept {
      auto propNo = propertyIndex(propName);
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (propNo == _N) {
        printf(
            "tilevector [%s] ofb! (pack) accessing prop [%s] which is not among %d props (%d "
            "chns, %lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_dims._numChannels,
            (long long int)_dims.size());
        return static_cast<const base_t &>(*this).pack(dim_c<Ns...>, _dims._numChannels, i,
                                                       wrapt<TT>{});
        // using RetT = decltype(static_cast<const base_t &>(*this).pack(
        //     dim_c<Ns...>, _tagOffsets[propertyIndex(propName)], i, wrapt<TT>{}));
        // return RetT::zeros();
      }
#endif
      return static_cast<const base_t &>(*this).pack(dim_c<Ns...>, _tagOffsets[propNo], i,
                                                     wrapt<TT>{});
    }
    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto pack(const SmallString &propName, const size_type i,
                        wrapt<TT> = {}) const noexcept {
      return pack(dim_c<Ns...>, propName, i, wrapt<TT>{});
    }

    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto pack(value_seq<Ns...>, const SmallString &propName,
                        const channel_counter_type chn, const size_type i,
                        wrapt<TT> = {}) const noexcept {
      auto propNo = propertyIndex(propName);
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (propNo == _N) {
        printf(
            "tilevector [%s] ofb! (pack) accessing prop [%s] which is not among %d props (%d "
            "chns, "
            "%lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_dims._numChannels,
            (long long int)_dims.size());
        return static_cast<const base_t &>(*this).pack(dim_c<Ns...>, _dims._numChannels, i,
                                                       wrapt<TT>{});
        // using RetT = decltype(static_cast<const base_t &>(*this).pack(
        //     dim_c<Ns...>, _tagOffsets[propertyIndex(propName)] + chn, i, wrapt<TT>{}));
        // return RetT::zeros();
      }
#endif
      return static_cast<const base_t &>(*this).pack(dim_c<Ns...>, _tagOffsets[propNo] + chn, i,
                                                     wrapt<TT>{});
    }
    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto pack(const SmallString &propName, const channel_counter_type chn,
                        const size_type i, wrapt<TT> = {}) const noexcept {
      return pack(dim_c<Ns...>, propName, chn, i, wrapt<TT>{});
    }

    template <channel_counter_type N, typename VT = value_type>
    constexpr auto array(const SmallString &propName, const size_type i) const noexcept {
      auto propNo = propertyIndex(propName);
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (propNo == _N) {
        printf(
            "tilevector [%s] ofb! (array) accessing prop [%s] which is not among %d props (%d "
            "chns, %lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_dims._numChannels,
            (long long int)_dims.size());
        return static_cast<const base_t &>(*this).template array<N, VT>(_dims._numChannels, i);
      }
#endif
      return static_cast<const base_t &>(*this).template array<N, VT>(_tagOffsets[propNo], i);
    }

    /// @brief tieing
    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto tuple(value_seq<Ns...>, const SmallString &propName, const size_type i,
                         wrapt<TT> = {}) const noexcept {
      auto propNo = propertyIndex(propName);
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (propNo == _N) {
        printf(
            "tilevector [%s] ofb! (tuple) const accessing prop [%s] which is not among %d props "
            "(%d chns, %lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_dims._numChannels,
            (long long int)_dims.size());
        return static_cast<const base_t &>(*this).tuple(dim_c<Ns...>, _dims._numChannels, i,
                                                        wrapt<TT>{});
      }
#endif
      return static_cast<const base_t &>(*this).tuple(dim_c<Ns...>, _tagOffsets[propNo], i,
                                                      wrapt<TT>{});
    }
    template <auto d, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto tuple(const SmallString &propName, const size_type i,
                         wrapt<TT> = {}) const noexcept {
      return tuple(dim_c<d>, propName, i, wrapt<TT>{});
    }

    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto tuple(value_seq<Ns...>, const SmallString &propName,
                         const channel_counter_type chn, const size_type i,
                         wrapt<TT> = {}) const noexcept {
      auto propNo = propertyIndex(propName);
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (propNo == _N) {
        printf(
            "tilevector [%s] ofb! (tuple) const accessing prop [%s] which is not among %d props "
            "(%d chns, %lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_dims._numChannels,
            (long long int)_dims.size());
        return static_cast<const base_t &>(*this).tuple(dim_c<Ns...>, _dims._numChannels, i,
                                                        wrapt<TT>{});
      }
#endif
      return static_cast<const base_t &>(*this).tuple(dim_c<Ns...>, _tagOffsets[propNo] + chn, i,
                                                      wrapt<TT>{});
    }
    template <auto d, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto tuple(const SmallString &propName, const channel_counter_type chn,
                         const size_type i, wrapt<TT> = {}) const noexcept {
      return tuple(dim_c<d>, propName, chn, i, wrapt<TT>{});
    }

    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto mount(value_seq<Ns...>, const SmallString &propName, const size_type i,
                         wrapt<TT> = {}) const noexcept {
      auto propNo = propertyIndex(propName);
      return static_cast<const base_t &>(*this).mount(dim_c<Ns...>, _tagOffsets[propNo], i,
                                                      wrapt<TT>{});
    }
    template <auto d, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto mount(const SmallString &propName, const size_type i,
                         wrapt<TT> = {}) const noexcept {
      return mount(dim_c<d>, propName, i, wrapt<TT>{});
    }

    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto mount(value_seq<Ns...>, const SmallString &propName,
                         const channel_counter_type chn, const size_type i,
                         wrapt<TT> = {}) const noexcept {
      auto propNo = propertyIndex(propName);
      return static_cast<const base_t &>(*this).mount(dim_c<Ns...>, _tagOffsets[propNo] + chn, i,
                                                      wrapt<TT>{});
    }
    template <auto d, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (alignof(TT) == alignof(value_type))>
              = 0>
    constexpr auto mount(const SmallString &propName, const channel_counter_type chn,
                         const size_type i, wrapt<TT> = {}) const noexcept {
      return mount(dim_c<d>, propName, chn, i, wrapt<TT>{});
    }

    const SmallString *_tagNames{nullptr};
    const channel_counter_type *_tagOffsets{nullptr};
    const channel_counter_type *_tagSizes{nullptr};
    channel_counter_type _N{0};
  };

  template <execspace_e ExecSpace, typename T, size_t Length, typename Allocator,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(const std::vector<SmallString> &tagNames,
                      const TileVector<T, Length, Allocator> &vec, wrapv<Base> = {}) {
    return TileVectorView<ExecSpace, const TileVector<T, Length, Allocator>, false, Base>{{}, vec};
  }
  template <execspace_e ExecSpace, typename T, size_t Length, typename Allocator,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(const std::vector<SmallString> &tagNames,
                      TileVector<T, Length, Allocator> &vec, wrapv<Base> = {}) {
    return TileVectorView<ExecSpace, TileVector<T, Length, Allocator>, false, Base>{tagNames, vec};
  }

  /// tagged tilevector for debug
  template <execspace_e ExecSpace, typename T, size_t Length, typename Allocator,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(const std::vector<SmallString> &tagNames,
                      const TileVector<T, Length, Allocator> &vec, wrapv<Base>,
                      const SmallString &tagName) {
    auto ret = TileVectorView<ExecSpace, const TileVector<T, Length, Allocator>, false, Base>{
        tagNames, vec};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ret._nameTag = tagName;
#endif
    return ret;
  }
  template <execspace_e ExecSpace, typename T, size_t Length, typename Allocator,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(const std::vector<SmallString> &tagNames,
                      TileVector<T, Length, Allocator> &vec, wrapv<Base>,
                      const SmallString &tagName) {
    auto ret
        = TileVectorView<ExecSpace, TileVector<T, Length, Allocator>, false, Base>{tagNames, vec};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ret._nameTag = tagName;
#endif
    return ret;
  }

  template <execspace_e space, typename T, size_t Length, typename Allocator>
  decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                       const TileVector<T, Length, Allocator> &vec) {
    for (auto &&tag : tagNames)
      if (!vec.hasProperty(tag))
        throw std::runtime_error(
            fmt::format("tilevector property [\"{}\"] does not exist", (std::string)tag));
    return view<space>(tagNames, vec, false_c);
  }
  template <execspace_e space, typename T, size_t Length, typename Allocator>
  decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                       TileVector<T, Length, Allocator> &vec) {
    for (auto &&tag : tagNames)
      if (!vec.hasProperty(tag))
        throw std::runtime_error(
            fmt::format("tilevector property [\"{}\"] does not exist\n", (std::string)tag));
    return view<space>(tagNames, vec, false_c);
  }

  /// tagged tilevector for debug
  template <execspace_e space, typename T, size_t Length, typename Allocator>
  decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                       const TileVector<T, Length, Allocator> &vec, const SmallString &tagName) {
    for (auto &&tag : tagNames)
      if (!vec.hasProperty(tag))
        throw std::runtime_error(
            fmt::format("tilevector property [\"{}\"] does not exist\n", (std::string)tag));
    return view<space>(tagNames, vec, false_c, tagName);
  }
  template <execspace_e space, typename T, size_t Length, typename Allocator>
  decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                       TileVector<T, Length, Allocator> &vec, const SmallString &tagName) {
    for (auto &&tag : tagNames)
      if (!vec.hasProperty(tag))
        throw std::runtime_error(
            fmt::format("tilevector property [\"{}\"] does not exist\n", (std::string)tag));
    return view<space>(tagNames, vec, false_c, tagName);
  }

#if ZS_ENABLE_SERIALIZATION
  template <typename S, typename T, size_t Length>
  void serialize(S &s, TileVector<T, Length, ZSPmrAllocator<>> &c) {
    using C = TileVector<T, Length, ZSPmrAllocator<>>;
    if (!c.memoryLocation().onHost()) {
      c = c.clone({memsrc_e::host, -1});
    }

    s.template value<sizeof(typename C::channel_counter_type)>(c.refNumChannels());

    serialize(s, c.refBuffer());
    s.container(c.refTags(), detail::deduce_numeric_max<typename C::size_type>(),
                [](S &s, PropertyTag &v) { serialize(s, v); });
    serialize(s, c.refTagNames());
    serialize(s, c.refTagSizes());
    serialize(s, c.refTagOffsets());

    s.template value<sizeof(typename C::size_type)>(c.refSize());
  }
#endif

}  // namespace zs

#if ZS_ENABLE_SERIALIZATION
namespace bitsery {
  namespace traits {
    template <typename T> struct ContainerTraits;
    template <typename T> struct BufferAdapterTraits;

    template <typename T, size_t Length>
    struct ContainerTraits<zs::TileVector<T, Length, zs::ZSPmrAllocator<>>> {
      using container_type = zs::TileVector<T, Length, zs::ZSPmrAllocator<>>;
      using TValue = typename container_type::value_type;
      static constexpr bool isResizable = true;
      static constexpr bool isContiguous = true;
      static size_t size(const container_type &container) { return container.bufferSize(); }
      static void resize(container_type &container, size_t size) { container.resizeBuffer(size); }
    };
    template <typename T, size_t Length>
    struct BufferAdapterTraits<zs::TileVector<T, Length, zs::ZSPmrAllocator<>>> {
      using container_type = zs::TileVector<T, Length, zs::ZSPmrAllocator<>>;
      using TIterator = decltype(zs::declval<container_type &>().begin());
      using TConstIterator = decltype(zs::declval<const container_type &>().begin());
      using TValue = typename ContainerTraits<container_type>::TValue;
      static void increaseBufferSize(container_type &container, size_t /*currSize*/,
                                     size_t minSize) {
        const auto numElePerTile = container.numChannels() * container_type::lane_width;
        minSize = (minSize + numElePerTile - 1) / numElePerTile * numElePerTile;
        container.resizeBuffer(minSize);
      }
    };
  }  // namespace traits
}  // namespace bitsery
#endif
