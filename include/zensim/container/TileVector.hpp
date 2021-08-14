#pragma once
#include "Vector.hpp"
#include "zensim/math/Vec.h"
#include "zensim/memory/Allocator.h"
#include "zensim/resource/Resource.h"
#include "zensim/types/Iterator.h"
#include "zensim/types/SmallVector.hpp"

namespace zs {

  template <typename T, std::size_t Length = 8, typename Index = std::size_t,
            typename ChnCounter = unsigned char>
  struct TileVector {
    static_assert(is_same_v<T, remove_cvref_t<T>>, "T is not cvref-unqualified type!");
    static_assert(std::is_default_constructible_v<T> && std::is_trivially_copyable_v<T>,
                  "element is not default-constructible or trivially-copyable!");

    using value_type = T;
    using allocator_type = ZSPmrAllocator<>;
    using size_type = std::make_unsigned_t<Index>;
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
    constexpr decltype(auto) allocator() const noexcept { return _allocator; }

    TileVector(const allocator_type &allocator, const std::vector<PropertyTag> &channelTags,
               size_type count = 0)
        : _allocator{allocator},
          _base{nullptr},
          _tags{channelTags},
          _size{count},
          _capacity{count_tiles(count) * lane_width},
          _numChannels{numTotalChannels(channelTags)} {
      const auto N = numProperties();
      _base = (pointer)_allocator.allocate(sizeof(value_type) * numChannels() * capacity(),
                                           std::alignment_of_v<value_type>);
      {
        auto tagNames = Vector<SmallString>{static_cast<std::size_t>(N)};
        auto tagSizes = Vector<channel_counter_type>{static_cast<std::size_t>(N)};
        auto tagOffsets = Vector<channel_counter_type>{static_cast<std::size_t>(N)};
        channel_counter_type curOffset = 0;
        for (auto &&[name, size, offset, src] : zip(tagNames, tagSizes, tagOffsets, channelTags)) {
          name = zs::get<SmallString>(src);
          size = zs::get<1>(src);
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
        : TileVector{get_memory_source(mre, devid), channelTags, count} {}
    TileVector(channel_counter_type numChns, size_type count = 0, memsrc_e mre = memsrc_e::host,
               ProcID devid = -1)
        : TileVector{get_memory_source(mre, devid), {{"unnamed", numChns}}, count} {}
    TileVector(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : TileVector{get_memory_source(mre, devid), {{"unnamed", 1}}, 0} {}

    ~TileVector() {
      if (_base && capacity() > 0)
        _allocator.deallocate(_base, sizeof(value_type) * numChannels() * capacity(),
                              std::alignment_of_v<value_type>);
    }

    static auto numTotalChannels(const std::vector<PropertyTag> &tags) {
      channel_counter_type cnt = 0;
      for (std::size_t i = 0; i != tags.size(); ++i) cnt += tags[i].template get<1>();
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
          _base{(pointer)_allocator.allocate(sizeof(value_type) * o.numChannels() * o.capacity(),
                                             std::alignment_of_v<value_type>)},
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
      // capacity() is the count of tiles
      // use size() that represents the number of elements!
      TileVector ret{allocator, _tags, capacity()};
      copy(MemoryEntity{allocator.location, (void *)ret.data()},
           MemoryEntity{memoryLocation(), (void *)data()},
           sizeof(value_type) * numChannels() * capacity());
      return ret;
    }
    TileVector clone(const MemoryLocation &mloc) const {
      return clone(get_memory_source(mloc.memspace(), mloc.devid()));
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
        const auto newCapacity = geometric_size_growth(newSize);
        if (newCapacity > oldCapacity) {
          /// virtual memory way
          /// conventional way
          TileVector tmp{_allocator, _tags, newCapacity};
          if (size())
            copy(MemoryEntity{tmp.memoryLocation(), (void *)tmp.data()},
                 MemoryEntity{memoryLocation(), (void *)data()}, numTiles() * tileBytes());
          tmp._size = newSize;
          swap(tmp);
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
        if (str == zs::get<SmallString>(tag)) return true;
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
      auto tagOffsets = _tagOffsets.clone({memsrc_e::host, -1});
      for (auto &&[offset, tag] : zip(tagOffsets, _tags))
        if (str == zs::get<SmallString>(tag)) return offset;
      return 0;
    }
    constexpr PropertyTag getPropertyTag(std::size_t i = 0) const { return _tags[i]; }
    constexpr const auto &getPropertyTags() const { return _tags; }

  protected:
    allocator_type _allocator{};
    pointer _base{nullptr};
    std::vector<PropertyTag> _tags{};  // on host
    /// for proxy use
    Vector<SmallString> _tagNames{};
    Vector<channel_counter_type> _tagSizes{};
    Vector<channel_counter_type> _tagOffsets{};
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
  template <typename T, std::size_t Length, typename Index, typename ChnT>
  template <typename Policy>
  void TileVector<T, Length, Index, ChnT>::append_channels(Policy &&policy,
                                                           const std::vector<PropertyTag> &tags) {
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    const auto size = size();
    auto tags = getPropertyTags();
    tags.insert(std::end(tags), std::begin(tags), std::end(tags));
    TileVector<T, Length, Index, ChnT> tmp{allocator(), tags, size};
    policy(range(size), TileVectorCopy{proxy<space>(*this), proxy<space>(tmp)});
    *this = std::move(tmp);
  }

  template <execspace_e, typename TileVectorT, bool WithinTile = false, typename = void>
  struct TileVectorView;
  template <execspace_e, typename TileVectorT, bool WithinTile = false, typename = void>
  struct TileVectorUnnamedView;

  template <execspace_e Space, typename TileVectorT, bool WithinTile>
  struct TileVectorUnnamedView<Space, TileVectorT, WithinTile> {
    using pointer = typename TileVectorT::pointer;
    using value_type = typename TileVectorT::value_type;
    using reference = typename TileVectorT::reference;
    using const_reference = typename TileVectorT::const_reference;
    using size_type = typename TileVectorT::size_type;
    using channel_counter_type = typename TileVectorT::channel_counter_type;
    static constexpr auto lane_width = TileVectorT::lane_width;

    constexpr TileVectorUnnamedView() = default;
    ~TileVectorUnnamedView() = default;
    explicit constexpr TileVectorUnnamedView(TileVectorT &tilevector)
        : _vector{tilevector.data()},
          _vectorSize{tilevector.size()},
          _numChannels{tilevector.numChannels()} {}
    explicit constexpr TileVectorUnnamedView(pointer base, size_type s, channel_counter_type nchns)
        : _vector{base}, _vectorSize{s}, _numChannels{nchns} {}
    constexpr reference operator()(channel_counter_type chn, const size_type i) {
      if constexpr (WithinTile)
        return *(_vector + chn * lane_width + i);
      else
        return *(_vector + (i / lane_width * _numChannels + chn) * lane_width + i % lane_width);
    }
    constexpr const_reference operator()(channel_counter_type chn, const size_type i) const {
      if constexpr (WithinTile)
        return *(_vector + chn * lane_width + i);
      else
        return *(_vector + (i / lane_width * _numChannels + chn) * lane_width + i % lane_width);
    }

    constexpr auto tile(const size_type tileid) {
      return TileVectorUnnamedView<Space, TileVectorT, true>{
          _vector + tileid * lane_width * _numChannels, lane_width, _numChannels};
    }
    constexpr auto tile(const size_type tileid) const {
      return TileVectorUnnamedView<Space, const TileVectorT, true>{
          _vector + tileid * lane_width * _numChannels, lane_width, _numChannels};
    }

    template <auto... Ns> constexpr auto pack(channel_counter_type chn, const size_type i) const {
      using RetT = vec<value_type, Ns...>;
      RetT ret{};
      auto offset = (i / lane_width * _numChannels + chn) * lane_width + (i % lane_width);
      for (channel_counter_type d = 0; d < RetT::extent; ++d, offset += lane_width)
        ret.val(d) = *(_vector + offset);
      return ret;
    }
    template <std::size_t... Is> constexpr auto tuple_impl(const channel_counter_type chnOffset,
                                                           const size_type i, index_seq<Is...>) {
      const auto a = i / lane_width * _numChannels;
      const auto b = i % lane_width;
      return zs::forward_as_tuple(*(_vector + (a + (chnOffset + Is)) * lane_width + b)...);
    }
    template <std::size_t... Is> constexpr auto stdtuple_impl(const channel_counter_type chnOffset,
                                                              const size_type i, index_seq<Is...>) {
      const auto a = i / lane_width * _numChannels;
      const auto b = i % lane_width;
      return std::forward_as_tuple(*(_vector + (a + (chnOffset + Is)) * lane_width + b)...);
    }
    template <auto d> constexpr auto tuple(channel_counter_type chn, const size_type i) {
      return tuple_impl(chn, i, std::make_index_sequence<d>{});
    }
    template <auto d> constexpr auto stdtuple(channel_counter_type chn, const size_type i) {
      return stdtuple_impl(chn, i, std::make_index_sequence<d>{});
    }

    constexpr channel_counter_type numChannels() const noexcept { return _numChannels; }

    pointer _vector{nullptr};
    size_type _vectorSize{0};
    channel_counter_type _numChannels{0};
  };

  template <execspace_e Space, typename TileVectorT, bool WithinTile>
  struct TileVectorUnnamedView<Space, const TileVectorT, WithinTile> {
    using const_pointer = typename TileVectorT::const_pointer;
    using const_reference = typename TileVectorT::const_reference;
    using value_type = typename TileVectorT::value_type;
    using size_type = typename TileVectorT::size_type;
    using channel_counter_type = typename TileVectorT::channel_counter_type;
    static constexpr auto lane_width = TileVectorT::lane_width;

    constexpr TileVectorUnnamedView() = default;
    ~TileVectorUnnamedView() = default;
    explicit constexpr TileVectorUnnamedView(const TileVectorT &tilevector)
        : _vector{tilevector.data()},
          _vectorSize{tilevector.size()},
          _numChannels{tilevector.numChannels()} {}
    explicit constexpr TileVectorUnnamedView(const_pointer base, size_type s,
                                             channel_counter_type nchns)
        : _vector{base}, _vectorSize{s}, _numChannels{nchns} {}
    constexpr const_reference operator()(channel_counter_type chn, const size_type i) const {
      if constexpr (WithinTile)
        return *(_vector + chn * lane_width + i);
      else
        return *(_vector + (i / lane_width * _numChannels + chn) * lane_width + i % lane_width);
    }

    constexpr auto tile(const size_type tileid) const {
      return TileVectorUnnamedView<Space, const TileVectorT, true>{
          _vector + tileid * lane_width * _numChannels, lane_width, _numChannels};
    }

    template <auto... Ns> constexpr auto pack(channel_counter_type chn, const size_type i) const {
      using RetT = vec<value_type, Ns...>;
      RetT ret{};
      auto offset = (i / lane_width * _numChannels + chn) * lane_width + (i % lane_width);
      for (channel_counter_type d = 0; d < RetT::extent; ++d, offset += lane_width)
        ret.val(d) = *(_vector + offset);
      return ret;
    }
    template <std::size_t... Is> constexpr auto tuple_impl(const channel_counter_type chnOffset,
                                                           const size_type i,
                                                           index_seq<Is...>) const {
      const auto a = i / lane_width * _numChannels;
      const auto b = i % lane_width;
      return zs::forward_as_tuple(*(_vector + (a + (chnOffset + Is)) * lane_width + b)...);
    }
    template <std::size_t... Is> constexpr auto stdtuple_impl(const channel_counter_type chnOffset,
                                                              const size_type i,
                                                              index_seq<Is...>) const {
      const auto a = i / lane_width * _numChannels;
      const auto b = i % lane_width;
      return std::forward_as_tuple(*(_vector + (a + (chnOffset + Is)) * lane_width + b)...);
    }
    template <auto d> constexpr auto tuple(channel_counter_type chn, const size_type i) const {
      return tuple_impl(chn, i, std::make_index_sequence<d>{});
    }
    template <auto d> constexpr auto stdtuple(channel_counter_type chn, const size_type i) const {
      return stdtuple_impl(chn, i, std::make_index_sequence<d>{});
    }

    constexpr channel_counter_type numChannels() const noexcept { return _numChannels; }

    const_pointer _vector{nullptr};
    size_type _vectorSize{0};
    channel_counter_type _numChannels{0};
  };
  template <execspace_e ExecSpace, typename T, std::size_t Length, typename IndexT, typename ChnT>
  constexpr decltype(auto) proxy(const TileVector<T, Length, IndexT, ChnT> &vec) {
    return TileVectorUnnamedView<ExecSpace, const TileVector<T, Length, IndexT, ChnT>>{vec};
  }
  template <execspace_e ExecSpace, typename T, std::size_t Length, typename IndexT, typename ChnT>
  constexpr decltype(auto) proxy(TileVector<T, Length, IndexT, ChnT> &vec) {
    return TileVectorUnnamedView<ExecSpace, TileVector<T, Length, IndexT, ChnT>>{vec};
  }

  template <execspace_e Space, typename TileVectorT, bool WithinTile>
  struct TileVectorView<Space, TileVectorT, WithinTile>
      : TileVectorUnnamedView<Space, TileVectorT, WithinTile> {
    using base_t = TileVectorUnnamedView<Space, TileVectorT, WithinTile>;
    using pointer = typename TileVectorT::pointer;
    using value_type = typename TileVectorT::value_type;
    using reference = typename TileVectorT::reference;
    using const_reference = typename TileVectorT::const_reference;
    using size_type = typename TileVectorT::size_type;
    using channel_counter_type = typename TileVectorT::channel_counter_type;
    static constexpr auto lane_width = TileVectorT::lane_width;

    constexpr TileVectorView() = default;
    ~TileVectorView() = default;
    explicit constexpr TileVectorView(const std::vector<SmallString> &tagNames,
                                      TileVectorT &tilevector)
        : base_t{tilevector},
          _tagNames{tilevector.tagNameHandle()},
          _tagOffsets{tilevector.tagOffsetHandle()},
          _tagSizes{tilevector.tagSizeHandle()},
          _N{static_cast<channel_counter_type>(tagNames.size())} {}
    explicit constexpr TileVectorView(pointer base, size_type s, channel_counter_type nchns,
                                      const SmallString *tagNames,
                                      const channel_counter_type *tagOffsets,
                                      const channel_counter_type *tagSizes, channel_counter_type N)
        : base_t{base, s, nchns},
          _tagNames{tagNames},
          _tagOffsets{tagOffsets},
          _tagSizes{tagSizes},
          _N{N} {}

    constexpr auto propIndex(const char propName[]) const {
      channel_counter_type i = 0;
      for (; i < _N; ++i)
        if (_tagNames[i] == propName) break;
      return i;
    }
    constexpr bool hasProp(const char propName[]) const { return propIndex(propName) != _N; }

    constexpr reference operator()(const char propName[], const channel_counter_type chn,
                                   const size_type i) {
      return static_cast<base_t &>(*this)(_tagOffsets[propIndex(propName)] + chn, i);
    }
    constexpr const_reference operator()(const char propName[], const channel_counter_type chn,
                                         const size_type i) const {
      return static_cast<const base_t &>(*this)(_tagOffsets[propIndex(propName)] + chn, i);
    }
    constexpr reference operator()(const char propName[], const size_type i) {
      return static_cast<base_t &>(*this)(_tagOffsets[propIndex(propName)], i);
    }
    constexpr const_reference operator()(const char propName[], const size_type i) const {
      return static_cast<const base_t &>(*this)(_tagOffsets[propIndex(propName)], i);
    }
    constexpr auto tile(const size_type tileid) {
      return TileVectorView<Space, TileVectorT, true>{
          this->_vector + tileid * lane_width * this->_numChannels,
          lane_width,
          this->_numChannels,
          _tagNames,
          _tagOffsets,
          _tagSizes,
          _N};
    }
    constexpr auto tile(const size_type tileid) const {
      return TileVectorView<Space, const TileVectorT, true>{
          this->_vector + tileid * lane_width * this->_numChannels,
          lane_width,
          this->_numChannels,
          _tagNames,
          _tagOffsets,
          _tagSizes,
          _N};
    }

    template <auto... Ns> constexpr auto pack(const char propName[], const size_type i) const {
      return static_cast<const base_t &>(*this).template pack<Ns...>(
          _tagOffsets[propIndex(propName)], i);
    }
    template <auto d> constexpr auto tuple(const char propName[], const size_type i) {
      return static_cast<base_t &>(*this).template tuple<d>(_tagOffsets[propIndex(propName)], i);
    }
    template <auto d> constexpr auto stdtuple(const char propName[], const size_type i) {
      return static_cast<base_t &>(*this).template stdtuple<d>(_tagOffsets[propIndex(propName)], i);
    }

    const SmallString *_tagNames{nullptr};
    const channel_counter_type *_tagOffsets{nullptr};
    const channel_counter_type *_tagSizes{nullptr};
    channel_counter_type _N{0};
  };
  template <execspace_e Space, typename TileVectorT, bool WithinTile>
  struct TileVectorView<Space, const TileVectorT, WithinTile>
      : TileVectorUnnamedView<Space, const TileVectorT, WithinTile> {
    using base_t = TileVectorUnnamedView<Space, const TileVectorT, WithinTile>;
    using const_pointer = typename TileVectorT::const_pointer;
    using value_type = typename TileVectorT::value_type;
    using const_reference = typename TileVectorT::const_reference;
    using size_type = typename TileVectorT::size_type;
    using channel_counter_type = typename TileVectorT::channel_counter_type;
    static constexpr auto lane_width = TileVectorT::lane_width;

    constexpr TileVectorView() = default;
    ~TileVectorView() = default;
    explicit constexpr TileVectorView(const std::vector<SmallString> &tagNames,
                                      const TileVectorT &tilevector)
        : base_t{tilevector},
          _tagNames{tilevector.tagNameHandle()},
          _tagOffsets{tilevector.tagOffsetHandle()},
          _tagSizes{tilevector.tagSizeHandle()},
          _N{static_cast<channel_counter_type>(tagNames.size())} {}
    explicit constexpr TileVectorView(const_pointer base, size_type s, channel_counter_type nchns,
                                      const SmallString *tagNames,
                                      const channel_counter_type *tagOffsets,
                                      const channel_counter_type *tagSizes, channel_counter_type N)
        : base_t{base, s, nchns},
          _tagNames{tagNames},
          _tagOffsets{tagOffsets},
          _tagSizes{tagSizes},
          _N{N} {}

    constexpr auto propIndex(const char propName[]) const {
      channel_counter_type i = 0;
      for (; i < _N; ++i)
        if (_tagNames[i] == propName) break;
      return i;
    }
    constexpr bool hasProp(const char propName[]) const { return propIndex(propName) != _N; }

    constexpr const_reference operator()(const char propName[], const channel_counter_type chn,
                                         const size_type i) const {
      return static_cast<const base_t &>(*this)(_tagOffsets[propIndex(propName)] + chn, i);
    }
    constexpr const_reference operator()(const char propName[], const size_type i) const {
      return static_cast<const base_t &>(*this)(_tagOffsets[propIndex(propName)], i);
    }
    constexpr auto tile(const size_type tileid) const {
      return TileVectorView<Space, const TileVectorT, true>{
          this->_vector + tileid * lane_width * this->_numChannels,
          lane_width,
          this->_numChannels,
          _tagNames,
          _tagOffsets,
          _tagSizes,
          _N};
    }

    template <auto... Ns> constexpr auto pack(const char propName[], const size_type i) const {
      return static_cast<const base_t &>(*this).template pack<Ns...>(
          _tagOffsets[propIndex(propName)], i);
    }
    template <auto d> constexpr auto tuple(const char propName[], const size_type i) {
      return static_cast<base_t &>(*this).template tuple<d>(_tagOffsets[propIndex(propName)], i);
    }
    template <auto d> constexpr auto stdtuple(const char propName[], const size_type i) {
      return static_cast<base_t &>(*this).template stdtuple<d>(_tagOffsets[propIndex(propName)], i);
    }

    const SmallString *_tagNames{nullptr};
    const channel_counter_type *_tagOffsets{nullptr};
    const channel_counter_type *_tagSizes{nullptr};
    channel_counter_type _N{0};
  };

  template <execspace_e ExecSpace, typename T, std::size_t Length, typename IndexT, typename ChnT>
  constexpr decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                                 const TileVector<T, Length, IndexT, ChnT> &vec) {
    for (auto &&tag : tagNames)
      if (!vec.hasProperty(tag))
        throw std::runtime_error(
            fmt::format("tilevector attribute [\"{}\"] not exists", (std::string)tag));
    return TileVectorView<ExecSpace, const TileVector<T, Length, IndexT, ChnT>>{tagNames, vec};
  }
  template <execspace_e ExecSpace, typename T, std::size_t Length, typename IndexT, typename ChnT>
  constexpr decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                                 TileVector<T, Length, IndexT, ChnT> &vec) {
    for (auto &&tag : tagNames)
      if (!vec.hasProperty(tag))
        throw std::runtime_error(
            fmt::format("tilevector attribute [\"{}\"] not exists\n", (std::string)tag));
    return TileVectorView<ExecSpace, TileVector<T, Length, IndexT, ChnT>>{tagNames, vec};
  }

}  // namespace zs