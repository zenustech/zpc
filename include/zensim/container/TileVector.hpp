#pragma once
#include "Vector.hpp"
#include "zensim/math/Vec.h"
#include "zensim/memory/Allocator.h"
#include "zensim/resource/Resource.h"
#include "zensim/types/Iterator.h"
#include "zensim/types/SmallVector.hpp"

namespace zs {

  template <typename T, std::size_t Length = 8, typename ChnCounter = unsigned char,
            typename AllocatorT = ZSPmrAllocator<>>
  struct TileVector {
    static_assert(is_zs_allocator<AllocatorT>::value,
                  "TileVector only works with zspmrallocator for now.");
    static_assert(is_same_v<T, remove_cvref_t<T>>, "T is not cvref-unqualified type!");
    static_assert(std::is_default_constructible_v<T> && std::is_trivially_copyable_v<T>,
                  "element is not default-constructible or trivially-copyable!");

    using value_type = T;
    using allocator_type = AllocatorT;
    using size_type = std::size_t;
    using difference_type = std::make_signed_t<size_type>;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    // tile vector specific configs
    using channel_counter_type = ChnCounter;
    static constexpr size_type lane_width = Length;

    static_assert(
        std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_swap::value,
        "allocator should propagate on copy, move and swap (for impl simplicity)!");

    static constexpr size_type count_tiles(size_type elementCount) noexcept {
      return (elementCount + lane_width - 1) / lane_width;
    }

    constexpr decltype(auto) memoryLocation() noexcept { return _allocator.location; }
    constexpr decltype(auto) memoryLocation() const noexcept { return _allocator.location; }
    constexpr ProcID devid() const noexcept { return memoryLocation().devid(); }
    constexpr memsrc_e memspace() const noexcept { return memoryLocation().memspace(); }
    decltype(auto) get_allocator() const noexcept { return _allocator; }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      if constexpr (is_virtual_zs_allocator<allocator_type>::value)
        return get_virtual_memory_source(mre, devid, (std::size_t)1 << (std::size_t)36, "STACK");
      else
        return get_memory_source(mre, devid);
    }
    pointer allocate(std::size_t bytes) {
      /// virtual memory way
      if constexpr (is_virtual_zs_allocator<allocator_type>::value) {
        _allocator.commit(0, bytes);
        return (pointer)_allocator.address(0);
      }
      /// conventional way
      else
        return (pointer)_allocator.allocate(bytes, std::alignment_of_v<value_type>);
    }

    TileVector(const allocator_type &allocator, const std::vector<PropertyTag> &channelTags,
               size_type count = 0)
        : _allocator{allocator},
          _base{nullptr},
          _tags{channelTags},
          _size{count},
          _capacity{count_tiles(count) * lane_width},
          _numChannels{numTotalChannels(channelTags)} {
      const auto N = numProperties();
      _base = allocate(sizeof(value_type) * numChannels() * capacity());
      {
        auto tagNames = Vector<SmallString, allocator_type>{static_cast<std::size_t>(N)};
        auto tagSizes = Vector<channel_counter_type, allocator_type>{static_cast<std::size_t>(N)};
        auto tagOffsets = Vector<channel_counter_type, allocator_type>{static_cast<std::size_t>(N)};
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

    ~TileVector() {
      if (_base && capacity() > 0)
        _allocator.deallocate(_base, sizeof(value_type) * numChannels() * capacity(),
                              std::alignment_of_v<value_type>);
    }

    static auto numTotalChannels(const std::vector<PropertyTag> &tags) {
      channel_counter_type cnt = 0;
      for (std::size_t i = 0; i != tags.size(); ++i) cnt += tags[i].numChannels;
      return cnt;
    }
    struct iterator_impl : IteratorInterface<iterator_impl> {
      constexpr iterator_impl(pointer base, size_type idx, channel_counter_type chn,
                              channel_counter_type nchns)
          : _base{base}, _idx{idx}, _chn{chn}, _numChannels{nchns} {}

      constexpr reference dereference() {
        return *(_base + (_idx / lane_width * _numChannels + _chn) * lane_width
                 + _idx % lane_width);
      }
      constexpr bool equal_to(iterator_impl it) const noexcept {
        return it._idx == _idx && it._chn == _chn;
      }
      constexpr void advance(difference_type offset) noexcept { _idx += offset; }
      constexpr difference_type distance_to(iterator_impl it) const noexcept {
        return it._idx - _idx;
      }

    protected:
      pointer _base{nullptr};
      size_type _idx{0};
      channel_counter_type _chn{0}, _numChannels{1};
    };
    struct const_iterator_impl : IteratorInterface<const_iterator_impl> {
      constexpr const_iterator_impl(const_pointer base, size_type idx, channel_counter_type chn,
                                    channel_counter_type nchns)
          : _base{base}, _idx{idx}, _chn{chn}, _numChannels{nchns} {}

      constexpr const_reference dereference() const {
        return *(_base + (_idx / lane_width * _numChannels + _chn) * lane_width
                 + _idx % lane_width);
      }
      constexpr bool equal_to(const_iterator_impl it) const noexcept {
        return it._idx == _idx && it._chn == _chn;
      }
      constexpr void advance(difference_type offset) noexcept { _idx += offset; }
      constexpr difference_type distance_to(const_iterator_impl it) const noexcept {
        return it._idx - _idx;
      }

    protected:
      const_pointer _base{nullptr};
      size_type _idx{0};
      channel_counter_type _chn{0}, _numChannels{1};
    };
    using iterator = LegacyIterator<iterator_impl>;
    using const_iterator = LegacyIterator<const_iterator_impl>;

    constexpr auto begin(channel_counter_type chn = 0) noexcept {
      return make_iterator<iterator_impl>(_base, 0, chn, numChannels());
    }
    constexpr auto end(channel_counter_type chn = 0) noexcept {
      return make_iterator<iterator_impl>(_base, size(), chn, numChannels());
    }
    constexpr auto begin(channel_counter_type chn = 0) const noexcept {
      return make_iterator<const_iterator_impl>(_base, 0, chn, numChannels());
    }
    constexpr auto end(channel_counter_type chn = 0) const noexcept {
      return make_iterator<const_iterator_impl>(_base, size(), chn, numChannels());
    }

    /// capacity
    constexpr size_type size() const noexcept { return _size; }
    constexpr size_type capacity() const noexcept { return _capacity; }
    constexpr channel_counter_type numChannels() const noexcept { return _numChannels; }
    constexpr size_type numTiles() const noexcept { return (size() + lane_width - 1) / lane_width; }
    constexpr size_type tileBytes() const noexcept {
      return numChannels() * lane_width * sizeof(value_type);
    }
    constexpr bool empty() noexcept { return size() == 0; }
    constexpr const_pointer data() const noexcept { return reinterpret_cast<const_pointer>(_base); }
    constexpr pointer data() noexcept { return reinterpret_cast<pointer>(_base); }

    /// element access
    constexpr reference operator[](
        const std::tuple<channel_counter_type, size_type> index) noexcept {
      const auto [chn, idx] = index;
      return *(data() + (idx / lane_width * numChannels() + chn) * lane_width + idx % lane_width);
    }
    constexpr conditional_t<std::is_fundamental_v<value_type>, value_type, const_reference>
    operator[](const std::tuple<channel_counter_type, size_type> index) const noexcept {
      const auto [chn, idx] = index;
      return *(data() + (idx / lane_width * numChannels() + chn) * lane_width + idx % lane_width);
    }
    /// ctor, assignment operator
    TileVector(const TileVector &o)
        : _allocator{o._allocator},
          _base{allocate(sizeof(value_type) * o.numChannels() * o.capacity())},
          _tags{o._tags},
          _tagNames{o._tagNames},
          _tagSizes{o._tagSizes},
          _tagOffsets{o._tagOffsets},
          _size{o.size()},
          _capacity{o.capacity()},
          _numChannels{o.numChannels()} {
      if (capacity() > 0)
        copy(MemoryEntity{memoryLocation(), (void *)data()},
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
      copy(MemoryEntity{allocator.location, (void *)ret.data()},
           MemoryEntity{memoryLocation(), (void *)data()},
           sizeof(value_type) * numChannels() * (count_tiles(size()) * lane_width));
      return ret;
    }
    TileVector clone(const MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }
    /// assignment or destruction after std::move
    /// https://www.youtube.com/watch?v=ZG59Bqo7qX4
    /// explicit noexcept
    /// leave the source object in a valid (default constructed) state
    TileVector(TileVector &&o) noexcept {
      const TileVector defaultVector{};
      _base = std::exchange(o._base, defaultVector._base);
      _allocator = std::exchange(o._allocator, defaultVector._allocator);
      _tags = std::exchange(o._tags, defaultVector._tags);
      _tagNames = std::exchange(o._tagNames, defaultVector._tagNames);
      _tagSizes = std::exchange(o._tagSizes, defaultVector._tagSizes);
      _tagOffsets = std::exchange(o._tagOffsets, defaultVector._tagOffsets);
      _size = std::exchange(o._size, defaultVector.size());
      _capacity = std::exchange(o._capacity, defaultVector.capacity());
      _numChannels = std::exchange(o._numChannels, defaultVector.numChannels());
    }
    /// make move-assignment safe for self-assignment
    TileVector &operator=(TileVector &&o) noexcept {
      if (this == &o) return *this;
      TileVector tmp(std::move(o));
      swap(tmp);
      return *this;
    }
    void swap(TileVector &o) noexcept {
      std::swap(_base, o._base);
      std::swap(_allocator, o._allocator);
      std::swap(_tags, o._tags);
      std::swap(_tagNames, o._tagNames);
      std::swap(_tagSizes, o._tagSizes);
      std::swap(_tagOffsets, o._tagOffsets);
      std::swap(_size, o._size);
      std::swap(_capacity, o._capacity);
      std::swap(_numChannels, o._numChannels);
    }
    friend void swap(TileVector &a, TileVector &b) { a.swap(b); }

    void clear() { *this = TileVector{_allocator, _tags, 0}; }
    void resize(size_type newSize) {
      const auto oldSize = size();
      if (newSize < oldSize) {
        _size = newSize;
        return;
      }
      if (newSize > oldSize) {
        const auto oldCapacity = capacity();
        if (newSize > oldCapacity) {
          /// virtual memory way
          if constexpr (is_virtual_zs_allocator<allocator_type>::value) {
            _capacity = count_tiles(geometric_size_growth(newSize)) * lane_width;
            _allocator.commit(capacity() * numChannels() * sizeof(value_type));
            _size = newSize;
          }
          /// conventional way
          else {
            TileVector tmp{_allocator, _tags, newSize};
            if (size())
              copy(MemoryEntity{tmp.memoryLocation(), (void *)tmp.data()},
                   MemoryEntity{memoryLocation(), (void *)data()}, numTiles() * tileBytes());
            swap(tmp);
          }
          return;
        }
      }
    }
    constexpr size_type geometric_size_growth(size_type newSize) noexcept {
      size_type geometricSize = capacity();
      geometricSize = geometricSize + geometricSize / 2;
      geometricSize = count_tiles(geometricSize) * lane_width;
      if (newSize > geometricSize) return count_tiles(newSize) * lane_width;
      return geometricSize;
    }
    template <typename Policy>
    void append_channels(Policy &&, const std::vector<PropertyTag> &tags);

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
    constexpr channel_counter_type getChannelOffset(const SmallString &str) const {
      channel_counter_type offset = 0;
      for (auto &&tag : _tags) {
        if (str == tag.name) return offset;
        offset += tag.numChannels;
      }
      return 0;
    }
    constexpr PropertyTag getPropertyTag(std::size_t i = 0) const { return _tags[i]; }
    constexpr const auto &getPropertyTags() const { return _tags; }

  protected:
    allocator_type _allocator{};
    pointer _base{nullptr};
    std::vector<PropertyTag> _tags{};  // on host
    /// for proxy use
    Vector<SmallString, allocator_type> _tagNames{};
    Vector<channel_counter_type, allocator_type> _tagSizes{};
    Vector<channel_counter_type, allocator_type> _tagOffsets{};
    size_type _size{0}, _capacity{0};  // element size
    channel_counter_type _numChannels{1};
  };

  template <typename TileVectorView> struct TileVectorCopy {
    using size_type = typename TileVectorView::size_type;
    using channel_counter_type = typename TileVectorView::channel_counter_type;
    TileVectorCopy(TileVectorView src, TileVectorView dst) : src{src}, dst{dst} {}
    constexpr void operator()(size_type i) {
      const auto nchns = src.numChannels();
      for (channel_counter_type chn = 0; chn != nchns; ++chn) dst(chn, i) = src(chn, i);
    }
    TileVectorView src, dst;
  };
  template <typename T, std::size_t Length, typename ChnT, typename Allocator>
  template <typename Policy> void TileVector<T, Length, ChnT, Allocator>::append_channels(
      Policy &&policy, const std::vector<PropertyTag> &appendTags) {
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    const auto s = size();
    auto tags = getPropertyTags();
    tags.insert(std::end(tags), std::begin(appendTags), std::end(appendTags));
    TileVector<T, Length, ChnT, Allocator> tmp{get_allocator(), tags, s};
    policy(range(s), TileVectorCopy{proxy<space>(*this), proxy<space>(tmp)});
    *this = std::move(tmp);
  }

  template <execspace_e Space, typename TileVectorT, bool WithinTile, typename = void>
  struct TileVectorUnnamedView {
    static constexpr bool is_const_structure = std::is_const_v<TileVectorT>;
    using tile_vector_type = std::remove_const_t<TileVectorT>;
    using const_tile_vector_type = std::add_const_t<tile_vector_type>;
    using pointer = typename tile_vector_type::pointer;
    using const_pointer = typename tile_vector_type::const_pointer;
    using value_type = typename tile_vector_type::value_type;
    using reference = typename tile_vector_type::reference;
    using const_reference = typename tile_vector_type::const_reference;
    using size_type = typename tile_vector_type::size_type;
    using difference_type = typename tile_vector_type::difference_type;
    using channel_counter_type = typename tile_vector_type::channel_counter_type;
    using whole_view_type = TileVectorUnnamedView<Space, TileVectorT, false>;
    using tile_view_type = TileVectorUnnamedView<Space, TileVectorT, true>;
    static constexpr auto lane_width = tile_vector_type::lane_width;
    static constexpr bool is_power_of_two = lane_width > 0 && (lane_width & (lane_width - 1)) == 0;
    static constexpr auto num_lane_bits = bit_count(lane_width);

    TileVectorUnnamedView() noexcept = default;
    ~TileVectorUnnamedView() noexcept = default;
    explicit constexpr TileVectorUnnamedView(TileVectorT &tilevector)
        : _vector{tilevector.data()},
          _vectorSize{tilevector.size()},
          _numChannels{tilevector.numChannels()} {}
    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    explicit constexpr TileVectorUnnamedView(pointer base, const size_type s,
                                             const channel_counter_type nchns)
        : _vector{base}, _vectorSize{s}, _numChannels{nchns} {}
    template <bool V = is_const_structure, enable_if_t<V> = 0>
    explicit constexpr TileVectorUnnamedView(const_pointer base, const size_type s,
                                             const channel_counter_type nchns)
        : _vector{base}, _vectorSize{s}, _numChannels{nchns} {}

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr reference operator()(const channel_counter_type chn, const size_type i) noexcept {
      if constexpr (WithinTile) {
        if constexpr (is_power_of_two)
          return *(_vector + ((chn << num_lane_bits) | i));
        else
          return *(_vector + (chn * lane_width + i));
      } else {
        if constexpr (is_power_of_two)
          return *(_vector
                   + ((((i >> num_lane_bits) * _numChannels + chn) << num_lane_bits)
                      | (i & (lane_width - 1))));
        else
          return *(_vector + (i / lane_width * _numChannels + chn) * lane_width + i % lane_width);
      }
    }
    constexpr const_reference operator()(const channel_counter_type chn,
                                         const size_type i) const noexcept {
      if constexpr (WithinTile) {
        if constexpr (is_power_of_two)
          return *(_vector + ((chn << num_lane_bits) | i));
        else
          return *(_vector + (chn * lane_width + i));
      } else {
        if constexpr (is_power_of_two)
          return *(_vector
                   + ((((i >> num_lane_bits) * _numChannels + chn) << num_lane_bits)
                      | (i & (lane_width - 1))));
        else
          return *(_vector + (i / lane_width * _numChannels + chn) * lane_width + i % lane_width);
      }
    }

    template <bool V = is_const_structure, bool InTile = WithinTile, enable_if_all<!V, !InTile> = 0>
    constexpr auto tile(const size_type tileid) noexcept {
      if constexpr (is_power_of_two)
        return TileVectorUnnamedView<Space, tile_vector_type, true>{
            _vector + (tileid << num_lane_bits) * _numChannels, lane_width, _numChannels};
      else
        return TileVectorUnnamedView<Space, tile_vector_type, true>{
            _vector + tileid * lane_width * _numChannels, lane_width, _numChannels};
    }
    template <bool InTile = WithinTile, enable_if_t<!InTile> = 0>
    constexpr auto tile(const size_type tileid) const noexcept {
      if constexpr (is_power_of_two)
        return TileVectorUnnamedView<Space, const_tile_vector_type, true>{
            _vector + (tileid << num_lane_bits) * _numChannels, lane_width, _numChannels};
      else
        return TileVectorUnnamedView<Space, const_tile_vector_type, true>{
            _vector + tileid * lane_width * _numChannels, lane_width, _numChannels};
    }

    template <auto... Ns>
    constexpr auto pack(channel_counter_type chn, const size_type i) const noexcept {
      using RetT = vec<value_type, Ns...>;
      RetT ret{};
      size_type offset{};
      if constexpr (is_power_of_two)
        offset = ((((i >> num_lane_bits) * _numChannels) + chn) << num_lane_bits)
                 | (i & (lane_width - 1));
      else
        offset = (i / lane_width * _numChannels + chn) * lane_width + (i % lane_width);
      for (channel_counter_type d = 0; d != RetT::extent; ++d, offset += lane_width)
        ret.val(d) = *(_vector + offset);
      return ret;
    }
    /// tuple
    template <std::size_t... Is, bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr auto tuple_impl(const channel_counter_type chnOffset, const size_type i,
                              index_seq<Is...>) noexcept {
      size_type a{}, b{};
      if constexpr (is_power_of_two) {
        a = (i >> num_lane_bits) * _numChannels;
        b = i & (lane_width - 1);
      } else {
        a = i / lane_width * _numChannels;
        b = i % lane_width;
      }
      if constexpr (is_power_of_two)
        return zs::tie(*(_vector + (((a + (chnOffset + Is)) << num_lane_bits) | b))...);
      else
        return zs::tie(*(_vector + (a + (chnOffset + Is)) * lane_width + b)...);
    }
    template <std::size_t... Is, bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr auto stdtuple_impl(const channel_counter_type chnOffset, const size_type i,
                                 index_seq<Is...>) noexcept {
      size_type a{}, b{};
      if constexpr (is_power_of_two) {
        a = (i >> num_lane_bits) * _numChannels;
        b = i & (lane_width - 1);
      } else {
        a = i / lane_width * _numChannels;
        b = i % lane_width;
      }
      if constexpr (is_power_of_two)
        return std::tie(*(_vector + (((a + (chnOffset + Is)) << num_lane_bits) | b))...);
      else
        return std::tie(*(_vector + (a + (chnOffset + Is)) * lane_width + b)...);
    }
    template <auto d, bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr auto tuple(const channel_counter_type chn, const size_type i) noexcept {
      return tuple_impl(chn, i, std::make_index_sequence<d>{});
    }
    template <auto d, bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr auto stdtuple(const channel_counter_type chn, const size_type i) noexcept {
      return stdtuple_impl(chn, i, std::make_index_sequence<d>{});
    }
    template <std::size_t... Is> constexpr auto tuple_impl(const channel_counter_type chnOffset,
                                                           const size_type i,
                                                           index_seq<Is...>) const noexcept {
      size_type a{}, b{};
      if constexpr (is_power_of_two) {
        a = (i >> num_lane_bits) * _numChannels;
        b = i & (lane_width - 1);
      } else {
        a = i / lane_width * _numChannels;
        b = i % lane_width;
      }
      if constexpr (is_power_of_two)
        return zs::tie(*(_vector + (((a + (chnOffset + Is)) << num_lane_bits) | b))...);
      else
        return zs::tie(*(_vector + (a + (chnOffset + Is)) * lane_width + b)...);
    }
    template <std::size_t... Is> constexpr auto stdtuple_impl(const channel_counter_type chnOffset,
                                                              const size_type i,
                                                              index_seq<Is...>) const noexcept {
      size_type a{}, b{};
      if constexpr (is_power_of_two) {
        a = (i >> num_lane_bits) * _numChannels;
        b = i & (lane_width - 1);
      } else {
        a = i / lane_width * _numChannels;
        b = i % lane_width;
      }
      if constexpr (is_power_of_two)
        return std::tie(*(_vector + (((a + (chnOffset + Is)) << num_lane_bits) | b))...);
      else
        return std::tie(*(_vector + (a + (chnOffset + Is)) * lane_width + b)...);
    }
    template <auto d>
    constexpr auto tuple(const channel_counter_type chn, const size_type i) const noexcept {
      return tuple_impl(chn, i, std::make_index_sequence<d>{});
    }
    template <auto d>
    constexpr auto stdtuple(const channel_counter_type chn, const size_type i) const noexcept {
      return stdtuple_impl(chn, i, std::make_index_sequence<d>{});
    }

    constexpr size_type size() const noexcept {
      if constexpr (WithinTile)
        return lane_width;
      else
        return _vectorSize;
    }
    constexpr channel_counter_type numChannels() const noexcept { return _numChannels; }

    conditional_t<is_const_structure, const_pointer, pointer> const _vector{nullptr};
    const size_type _vectorSize{0};
    const channel_counter_type _numChannels{0};
  };

  template <execspace_e ExecSpace, typename T, std::size_t Length, typename ChnT,
            typename Allocator>
  constexpr decltype(auto) proxy(const TileVector<T, Length, ChnT, Allocator> &vec) {
    return TileVectorUnnamedView<ExecSpace, const TileVector<T, Length, ChnT, Allocator>, false>{
        vec};
  }
  template <execspace_e ExecSpace, typename T, std::size_t Length, typename ChnT,
            typename Allocator>
  constexpr decltype(auto) proxy(TileVector<T, Length, ChnT, Allocator> &vec) {
    return TileVectorUnnamedView<ExecSpace, TileVector<T, Length, ChnT, Allocator>, false>{vec};
  }

  template <execspace_e Space, typename TileVectorT, bool WithinTile, typename = void>
  struct TileVectorView : TileVectorUnnamedView<Space, TileVectorT, WithinTile> {
    using base_t = TileVectorUnnamedView<Space, TileVectorT, WithinTile>;

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
    using whole_view_type = TileVectorView<Space, TileVectorT, false>;
    using tile_view_type = TileVectorView<Space, TileVectorT, true>;
    static constexpr auto lane_width = base_t::lane_width;

    TileVectorView() noexcept = default;
    ~TileVectorView() noexcept = default;
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

    constexpr auto propertyIndex(const SmallString &propName) const noexcept {
      channel_counter_type i = 0;
      for (; i != _N; ++i)
        if (_tagNames[i] == propName) break;
      return i;
    }
    constexpr bool hasProperty(const SmallString &propName) const noexcept {
      return propertyIndex(propName) != _N;
    }

    using base_t::operator();
    using base_t::pack;
    using base_t::stdtuple;
    using base_t::tuple;
    ///
    /// have to make sure that char type (might be channel_counter_type) not fit into this overload
    ///
    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr reference operator()(const SmallString &propName, const channel_counter_type chn,
                                   const size_type i) noexcept {
      return static_cast<base_t &>(*this)(_tagOffsets[propertyIndex(propName)] + chn, i);
    }
    constexpr const_reference operator()(const SmallString &propName,
                                         const channel_counter_type chn,
                                         const size_type i) const noexcept {
      return static_cast<const base_t &>(*this)(_tagOffsets[propertyIndex(propName)] + chn, i);
    }
    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr reference operator()(const SmallString &propName, const size_type i) noexcept {
      return static_cast<base_t &>(*this)(_tagOffsets[propertyIndex(propName)], i);
    }
    constexpr const_reference operator()(const SmallString &propName,
                                         const size_type i) const noexcept {
      return static_cast<const base_t &>(*this)(_tagOffsets[propertyIndex(propName)], i);
    }
    template <bool V = is_const_structure, bool InTile = WithinTile, enable_if_all<!V, !InTile> = 0>
    constexpr auto tile(const size_type tileid) noexcept {
      return TileVectorView<Space, TileVectorT, true>{
          this->_vector + tileid * lane_width * this->_numChannels,
          lane_width,
          this->_numChannels,
          _tagNames,
          _tagOffsets,
          _tagSizes,
          _N};
    }
    template <bool InTile = WithinTile, enable_if_t<!InTile> = 0>
    constexpr auto tile(const size_type tileid) const noexcept {
      return TileVectorView<Space, const_tile_vector_type, true>{
          this->_vector + tileid * lane_width * this->_numChannels,
          lane_width,
          this->_numChannels,
          _tagNames,
          _tagOffsets,
          _tagSizes,
          _N};
    }

    template <auto... Ns>
    constexpr auto pack(const SmallString &propName, const size_type i) const noexcept {
      return static_cast<const base_t &>(*this).template pack<Ns...>(
          _tagOffsets[propertyIndex(propName)], i);
    }
    template <auto d, bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr auto tuple(const SmallString &propName, const size_type i) noexcept {
      return static_cast<base_t &>(*this).template tuple<d>(_tagOffsets[propertyIndex(propName)],
                                                            i);
    }
    template <auto d, bool V = is_const_structure, enable_if_t<!V> = 0>
    constexpr auto stdtuple(const SmallString &propName, const size_type i) noexcept {
      return static_cast<base_t &>(*this).template stdtuple<d>(_tagOffsets[propertyIndex(propName)],
                                                               i);
    }
    template <auto d>
    constexpr auto tuple(const SmallString &propName, const size_type i) const noexcept {
      return static_cast<const base_t &>(*this).template tuple<d>(
          _tagOffsets[propertyIndex(propName)], i);
    }
    template <auto d>
    constexpr auto stdtuple(const SmallString &propName, const size_type i) const noexcept {
      return static_cast<const base_t &>(*this).template stdtuple<d>(
          _tagOffsets[propertyIndex(propName)], i);
    }

    const SmallString *_tagNames{nullptr};
    const channel_counter_type *_tagOffsets{nullptr};
    const channel_counter_type *_tagSizes{nullptr};
    const channel_counter_type _N{0};
  };

  template <execspace_e ExecSpace, typename T, std::size_t Length, typename ChnT,
            typename Allocator>
  constexpr decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                                 const TileVector<T, Length, ChnT, Allocator> &vec) {
    for (auto &&tag : tagNames)
      if (!vec.hasProperty(tag))
        throw std::runtime_error(
            fmt::format("tilevector attribute [\"{}\"] not exists", (std::string)tag));
    return TileVectorView<ExecSpace, const TileVector<T, Length, ChnT, Allocator>, false>{tagNames,
                                                                                          vec};
  }
  template <execspace_e ExecSpace, typename T, std::size_t Length, typename ChnT,
            typename Allocator>
  constexpr decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                                 TileVector<T, Length, ChnT, Allocator> &vec) {
    for (auto &&tag : tagNames)
      if (!vec.hasProperty(tag))
        throw std::runtime_error(
            fmt::format("tilevector attribute [\"{}\"] not exists\n", (std::string)tag));
    return TileVectorView<ExecSpace, TileVector<T, Length, ChnT, Allocator>, false>{tagNames, vec};
  }

}  // namespace zs