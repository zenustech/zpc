#pragma once
#include <type_traits>

#include "Vector.hpp"
#include "zensim/math/Vec.h"
#include "zensim/resource/Resource.h"
#include "zensim/types/Iterator.h"
#include "zensim/types/SmallVector.hpp"

namespace zs {

  template <typename T_, std::size_t Length = 8, typename AllocatorT = ZSPmrAllocator<>>
  struct TileVector {
    static_assert(is_zs_allocator<AllocatorT>::value,
                  "TileVector only works with zspmrallocator for now.");
    static_assert(is_same_v<T_, remove_cvref_t<T_>>, "T is not cvref-unqualified type!");
    static_assert(std::is_default_constructible_v<T_> && std::is_trivially_copyable_v<T_>,
                  "element is not default-constructible or trivially-copyable!");

    using value_type = T_;
    using allocator_type = AllocatorT;
    using size_type = std::size_t;
    using difference_type = std::make_signed_t<size_type>;
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

    template <typename T = value_type, typename Dims = value_seq<1>, typename Pred = void>
    struct iterator_impl : IteratorInterface<iterator_impl<T, Dims, Pred>> {
      static_assert(
          is_same_v<T, value_type>,
          "default iterator implementation only supports original \'value_type\' access.");
      static_assert(is_same_v<Dims, value_seq<1>>,
                    "default iterator implementation only supports scalar access.");

      static constexpr bool is_const_structure = std::is_const_v<T>;
      constexpr iterator_impl(pointer base, size_type idx, channel_counter_type chn,
                              channel_counter_type nchns) noexcept
          : _base{base}, _idx{idx}, _chn{chn}, _numChannels{nchns} {}

      template <auto V = is_const_structure, enable_if_t<!V> = 0>
      constexpr reference dereference() {
        return *(_base + (_idx / lane_width * _numChannels + _chn) * lane_width
                 + _idx % lane_width);
      }
      constexpr const_reference dereference() const {
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
      conditional_t<is_const_structure, const_pointer, pointer> _base{nullptr};
      size_type _idx{0};
      channel_counter_type _chn{0}, _numChannels{1};
    };
    template <typename T = value_type, typename Dims = value_seq<1>> using iterator
        = LegacyIterator<iterator_impl<T, Dims>>;
    template <typename T = const value_type, typename Dims = value_seq<1>> using const_iterator
        = LegacyIterator<iterator_impl<std::add_const_t<T>, Dims>>;

    template <typename T = value_type, typename Dims = value_seq<1>>
    constexpr auto begin(channel_counter_type chn = 0, wrapt<T> = {}, Dims = {}) noexcept {
      return make_iterator<iterator_impl<T, Dims>>(_base, static_cast<size_type>(0), chn,
                                                   numChannels());
    }
    template <typename T = value_type, typename Dims = value_seq<1>>
    constexpr auto end(channel_counter_type chn = 0, wrapt<T> = {}, Dims = {}) noexcept {
      return make_iterator<iterator_impl<T, Dims>>(_base, size(), chn, numChannels());
    }
    template <typename T = const value_type, typename Dims = value_seq<1>>
    constexpr auto begin(channel_counter_type chn = 0, wrapt<T> = {}, Dims = {}) const noexcept {
      return make_iterator<iterator_impl<std::add_const_t<T>, Dims>>(
          _base, static_cast<size_type>(0), chn, numChannels());
    }
    template <typename T = const value_type, typename Dims = value_seq<1>>
    constexpr auto end(channel_counter_type chn = 0, wrapt<T> = {}, Dims = {}) const noexcept {
      return make_iterator<iterator_impl<std::add_const_t<T>, Dims>>(_base, size(), chn,
                                                                     numChannels());
    }

    /// capacity
    constexpr size_type size() const noexcept { return _size; }
    constexpr size_type capacity() const noexcept { return _capacity; }
    constexpr channel_counter_type numChannels() const noexcept { return _numChannels; }
    constexpr size_type numTiles() const noexcept { return (size() + lane_width - 1) / lane_width; }
    constexpr size_type numReservedTiles() const noexcept {
      return (capacity() + lane_width - 1) / lane_width;
    }
    constexpr size_type tileBytes() const noexcept {
      return numChannels() * lane_width * sizeof(value_type);
    }
    constexpr bool empty() noexcept { return size() == 0; }
    constexpr const_pointer data() const noexcept { return reinterpret_cast<const_pointer>(_base); }
    constexpr pointer data() noexcept { return reinterpret_cast<pointer>(_base); }

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
    friend void swap(TileVector &a, TileVector &b) noexcept { a.swap(b); }

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
            TileVector tmp{_allocator, _tags, geometric_size_growth(newSize)};
            if (size())
              Resource::copy(MemoryEntity{tmp.memoryLocation(), (void *)tmp.data()},
                             MemoryEntity{memoryLocation(), (void *)data()},
                             numTiles() * tileBytes());
            tmp._size = newSize;
            swap(tmp);
          }
          return;
        } else
          _size = newSize;
      }
    }
    template <typename Policy> void reset(Policy &&policy, value_type val);
    void reset(int ch) {
      Resource::memset(MemoryEntity{memoryLocation(), (void *)data()}, ch,
                       numTiles() * tileBytes());
    }

    constexpr size_type geometric_size_growth(size_type newSize,
                                              size_type capacity) const noexcept {
      size_type geometricSize = capacity;
      geometricSize = geometricSize + geometricSize / 2;
      geometricSize = count_tiles(geometricSize) * lane_width;
      if (newSize > geometricSize) return count_tiles(newSize) * lane_width;
      return geometricSize;
    }
    constexpr size_type geometric_size_growth(size_type newSize) const noexcept {
      return geometric_size_growth(newSize, capacity());
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

#define EXTERN_TILEVECTOR_INSTANTIATIONS(LENGTH)                        \
  extern template struct TileVector<u32, LENGTH, ZSPmrAllocator<>>;     \
  extern template struct TileVector<u64, LENGTH, ZSPmrAllocator<>>;     \
  extern template struct TileVector<i32, LENGTH, ZSPmrAllocator<>>;     \
  extern template struct TileVector<i64, LENGTH, ZSPmrAllocator<>>;     \
  extern template struct TileVector<f32, LENGTH, ZSPmrAllocator<>>;     \
  extern template struct TileVector<f64, LENGTH, ZSPmrAllocator<>>;     \
  extern template struct TileVector<u32, LENGTH, ZSPmrAllocator<true>>; \
  extern template struct TileVector<u64, LENGTH, ZSPmrAllocator<true>>; \
  extern template struct TileVector<i32, LENGTH, ZSPmrAllocator<true>>; \
  extern template struct TileVector<i64, LENGTH, ZSPmrAllocator<true>>; \
  extern template struct TileVector<f32, LENGTH, ZSPmrAllocator<true>>; \
  extern template struct TileVector<f64, LENGTH, ZSPmrAllocator<true>>;

  /// 8, 32, 64, 512
  EXTERN_TILEVECTOR_INSTANTIATIONS(8)
  EXTERN_TILEVECTOR_INSTANTIATIONS(32)
  EXTERN_TILEVECTOR_INSTANTIATIONS(64)
  EXTERN_TILEVECTOR_INSTANTIATIONS(512)

  /// @brief tilevector iterator specializations
  template <typename T, std::size_t L, typename Allocator> template <typename ValT, auto... Ns>
  struct TileVector<T, L, Allocator>::iterator_impl<ValT, value_seq<Ns...>,
                                                    std::enable_if_t<sizeof(T) == sizeof(ValT)>>
      : IteratorInterface<typename TileVector<T, L, Allocator>::template iterator_impl<
            ValT, value_seq<Ns...>, std::enable_if_t<sizeof(T) == sizeof(ValT)>>> {
    static_assert(!std::is_reference_v<ValT>,
                  "the access type of the iterator should not be a reference.");
    static constexpr auto extent = (Ns * ...);
    static_assert(extent > 0, "the access extent of the iterator should be positive.");

    using iter_value_type = remove_cvref_t<ValT>;
    static constexpr bool is_const_structure = std::is_const_v<ValT>;
    static constexpr bool is_native_value_type = is_same_v<value_type, iter_value_type>;
    static constexpr bool is_scalar_access = (extent == 1);

    constexpr iterator_impl(conditional_t<is_const_structure, const_pointer, pointer> base,
                            size_type idx, channel_counter_type chn,
                            channel_counter_type nchns) noexcept
        : _base{base}, _idx{idx}, _chn{chn}, _numChannels{nchns} {}

    template <typename VecT, enable_if_t<is_vec<remove_cvref_t<VecT>>::value> = 0>
    static constexpr decltype(auto) interpret(VecT &&v) noexcept {
      if constexpr (is_native_value_type)
        return FWD(v);
      else
        return v.reinterpret_bits(wrapt<iter_value_type>{});
    }
    template <typename VT, enable_if_t<std::is_fundamental_v<remove_cvref_t<VT>>> = 0>
    static constexpr decltype(auto) interpret(VT &&v) noexcept {
      if constexpr (is_native_value_type)
        return FWD(v);
      else
        return reinterpret_bits<iter_value_type>(FWD(v));
    }

    constexpr decltype(auto) dereference() const {
      if constexpr (is_scalar_access)
        return interpret(
            *(_base + (_idx / lane_width * _numChannels + _chn) * lane_width + _idx % lane_width));
      else {
        using RetT = vec<value_type, Ns...>;
        RetT ret{};
        auto ptr
            = _base + (_idx / lane_width * _numChannels + _chn) * lane_width + (_idx % lane_width);
        for (channel_counter_type d = 0; d != extent; ++d, ptr += lane_width) ret.val(d) = *ptr;
        return interpret(ret);
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

  template <typename T, std::size_t L, typename Allocator> template <typename ValT, auto... Ns>
  struct TileVector<T, L, Allocator>::iterator_impl<
      ValT, value_seq<Ns...>,
      std::enable_if_t<(sizeof(T) * L % sizeof(ValT) == 0) && (sizeof(T) > sizeof(ValT))>>
      : IteratorInterface<typename TileVector<T, L, Allocator>::template iterator_impl<
            ValT, value_seq<Ns...>,
            std::enable_if_t<(sizeof(T) * L % sizeof(ValT) == 0) && (sizeof(T) > sizeof(ValT))>>> {
    static_assert(!std::is_reference_v<ValT>,
                  "the access type of the iterator should not be a reference.");
    static constexpr auto extent = (Ns * ...);
    static_assert(extent > 0, "the access extent of the iterator should be positive.");

    using iter_value_type = remove_cvref_t<ValT>;
    static constexpr bool is_const_structure = std::is_const_v<ValT>;
    static constexpr bool is_scalar_access = (extent == 1);

    static constexpr std::size_t num_segments
        = sizeof(value_type) * lane_width / sizeof(iter_value_type);

    constexpr iterator_impl(conditional_t<is_const_structure, const_pointer, pointer> base,
                            size_type idx, channel_counter_type segNo, channel_counter_type chn,
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
        using RetT = vec<iter_value_type, Ns...>;
        RetT ret{};
        auto ptr = (conditional_t<is_const_structure, const iter_value_type *,
                                  iter_value_type *>)(_base
                                                      + (_idx / lane_width * _numChannels + _chn)
                                                            * lane_width)
                   + _segOffset + _idx % lane_width;
        for (channel_counter_type d = 0; d != extent; ++d, ptr += lane_width * num_segments)
          ret.val(d) = *ptr;
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
    channel_counter_type _chn{0}, _numChannels{1}, _segOffset{0};
  };

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
  template <typename T, std::size_t Length, typename Allocator> template <typename Policy>
  void TileVector<T, Length, Allocator>::append_channels(
      Policy &&policy, const std::vector<PropertyTag> &appendTags) {
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    const auto s = size();
    auto tags = getPropertyTags();
    bool modified = false;
    for (const auto &tag : appendTags) {
      std::size_t i = 0;
      for (; i != tags.size(); ++i)
        if (tags[i].name == tag.name) break;
      if (i == tags.size()) {
        tags.push_back(tag);
        modified = true;
      } else if (tags[i].numChannels != tag.numChannels)
        throw std::runtime_error(
            fmt::format("append_channels: property[{}] currently has [{}] channels, cannot change "
                        "it to [{}] channels.",
                        tag.name, tags[i].numChannels, tag.numChannels));
    }
    if (!modified) return;
    TileVector<T, Length, Allocator> tmp{get_allocator(), tags, s};
    policy(range(s), TileVectorCopy{proxy<space>(*this), proxy<space>(tmp)});
    *this = std::move(tmp);
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
  template <typename T, std::size_t Length, typename Allocator> template <typename Policy>
  void TileVector<T, Length, Allocator>::reset(Policy &&policy, value_type val) {
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    policy(range(size()), TileVectorReset{proxy<space>(*this), val});
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

    TileVectorUnnamedView() noexcept = default;
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

    constexpr size_type numTiles() const noexcept {
      return (_vectorSize + lane_width - 1) / lane_width;
    }
    template <bool V = is_const_structure, typename TT = value_type,
              enable_if_all<!V, sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr std::add_lvalue_reference_t<TT> operator()(const channel_counter_type chn,
                                                         const size_type i,
                                                         wrapt<TT> = {}) noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if ((TT *)_vector == nullptr) {
        /// @note TBD : insert type reflection info here
        printf("tilevector [%s] operator() reinterpret_cast failed!\n", _nameTag.asChars());
        return *((TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
      }
      if (chn >= _numChannels) {
        printf("tilevector [%s] ofb! accessing chn [%d] out of [0, %d)\n", _nameTag.asChars(),
               (int)chn, (int)_numChannels);
        return *((TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
      }
#endif
      if constexpr (WithinTile) {
#if ZS_ENABLE_OFB_ACCESS_CHECK
        if (i >= lane_width) {
          printf("tilevector [%s] ofb! in-tile accessing ele [%lld] out of [0, %lld)\n",
                 _nameTag.asChars(), (long long)i, (long long)lane_width);
          return *((TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
        }
#endif

        return *((TT *)_vector + (chn * lane_width + i));
      } else {
#if ZS_ENABLE_OFB_ACCESS_CHECK
        if (i >= _vectorSize) {
          printf("tilevector [%s] ofb! global accessing ele [%lld] out of [0, %lld)\n",
                 _nameTag.asChars(), (long long)i, (long long)_vectorSize);
          return *((TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
        }
#endif
        return *((TT *)_vector
                 + ((i / lane_width * _numChannels + chn) * lane_width + i % lane_width));
      }
    }
    template <typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr TT operator()(const channel_counter_type chn, const size_type i,
                            wrapt<TT> = {}) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if ((const TT *)_vector == nullptr) {
        /// @note TBD : insert type reflection info here
        printf("tilevector [%s] operator() reinterpret_cast failed!\n", _nameTag.asChars());
        return *((const TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
      }
      if (chn >= _numChannels) {
        printf("tilevector [%s] ofb! accessing chn [%d] out of [0, %d)\n", _nameTag.asChars(),
               (int)chn, (int)_numChannels);
        return *((const TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
      }
#endif
      if constexpr (WithinTile) {
#if ZS_ENABLE_OFB_ACCESS_CHECK
        if (i >= lane_width) {
          printf("tilevector [%s] ofb! in-tile accessing ele [%lld] out of [0, %lld)\n",
                 _nameTag.asChars(), (long long)i, (long long)lane_width);
          return *((const TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
        }
#endif
        return *((const TT *)_vector + (chn * lane_width + i));
      } else {
#if ZS_ENABLE_OFB_ACCESS_CHECK
        if (i >= _vectorSize) {
          printf("tilevector [%s] ofb! global accessing ele [%lld] out of [0, %lld)\n",
                 _nameTag.asChars(), (long long)i, (long long)_vectorSize);
          return *((const TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
        }
#endif
        return *((const TT *)_vector
                 + ((i / lane_width * _numChannels + chn) * lane_width + i % lane_width));
      }
    }

    template <typename TT = value_type, bool V = is_const_structure, bool InTile = WithinTile,
              enable_if_all<!V, !InTile, sizeof(TT) == sizeof(value_type),
                            is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr std::add_lvalue_reference_t<TT> operator()(const channel_counter_type chn,
                                                         const size_type tileNo,
                                                         const size_type localNo,
                                                         wrapt<TT> = {}) noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if ((TT *)_vector == nullptr) {
        /// @note TBD : insert type reflection info here
        printf("tilevector [%s] operator()[tile] reinterpret_cast failed!\n", _nameTag.asChars());
        return *((TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
      }
      if (chn >= _numChannels) {
        printf("tilevector [%s] ofb! accessing chn [%d] out of [0, %d)\n", _nameTag.asChars(),
               (int)chn, (int)_numChannels);
        return *((TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
      }
      if (localNo >= lane_width) {
        printf("tilevector [%s] ofb! local accessing ele [%lld] out of [0, %lld)\n",
               _nameTag.asChars(), (long long)tileNo, (long long)lane_width);
        return *((TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
      }
#endif
      return *((TT *)_vector + ((tileNo * _numChannels + chn) * lane_width + localNo));
    }
    template <
        typename TT = value_type, bool InTile = WithinTile,
        enable_if_all<!InTile, sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                      (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr TT operator()(const channel_counter_type chn, const size_type tileNo,
                            const size_type localNo, wrapt<TT> = {}) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if ((const TT *)_vector == nullptr) {
        /// @note TBD : insert type reflection info here
        printf("tilevector [%s] operator()[tile] reinterpret_cast failed!\n", _nameTag.asChars());
        return *((const TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
      }
      if (chn >= _numChannels) {
        printf("tilevector [%s] ofb! accessing chn [%d] out of [0, %d)\n", _nameTag.asChars(),
               (int)chn, (int)_numChannels);
        return *((const TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
      }
      if (localNo >= lane_width) {
        printf("tilevector [%s] ofb! local accessing ele [%lld] out of [0, %lld)\n",
               _nameTag.asChars(), (long long)tileNo, (long long)lane_width);
        return *((const TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
      }
#endif
      return *((const TT *)_vector + ((tileNo * _numChannels + chn) * lane_width + localNo));
    }

    template <bool V = is_const_structure, bool InTile = WithinTile, enable_if_all<!V, !InTile> = 0>
    constexpr auto tile(const size_type tileid) noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (long long nt = numTiles(); tileid >= nt) {
        printf("tilevector [%s] ofb! global accessing tile [%lld] out of [0, %lld)\n",
               _nameTag.asChars(), (long long)tileid, nt);
        return TileVectorUnnamedView<Space, tile_vector_type, true>{(value_type *)0, lane_width,
                                                                    _numChannels};
      }
#endif
      return TileVectorUnnamedView<Space, tile_vector_type, true>{
          _vector + tileid * lane_width * _numChannels, lane_width, _numChannels};
    }
    template <bool InTile = WithinTile, enable_if_t<!InTile> = 0>
    constexpr auto tile(const size_type tileid) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (long long nt = numTiles(); tileid >= nt) {
        printf("tilevector [%s] ofb! global accessing tile [%lld] out of [0, %lld)\n",
               _nameTag.asChars(), (long long)tileid, nt);
        return TileVectorUnnamedView<Space, const_tile_vector_type, true>{(const value_type *)0,
                                                                          lane_width, _numChannels};
      }
#endif
      return TileVectorUnnamedView<Space, const_tile_vector_type, true>{
          _vector + tileid * lane_width * _numChannels, lane_width, _numChannels};
    }

    // use dim_c<Ns...> for the first parameter
    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr auto pack(value_seq<Ns...>, channel_counter_type chn, const size_type i,
                        wrapt<TT> = {}) const noexcept {
      using RetT = vec<TT, Ns...>;
      RetT ret{};
#if ZS_ENABLE_OFB_ACCESS_CHECK
      /// @brief check reinterpret_cast result validity (bit_cast should be more robust)
      if ((const TT *)_vector == nullptr) {
        /// @note TBD : insert type reflection info here
        printf("tilevector [%s] packing reinterpret_cast failed!\n", _nameTag.asChars());
        return RetT::constant(*(const TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
      }
      /// @brief check channel access overflow
      if (chn + RetT::extent > _numChannels) {
        printf("tilevector [%s] ofb! accessing chn [%d, %d) out of [0, %d)\n", _nameTag.asChars(),
               (int)chn, (int)(chn + RetT::extent), (int)_numChannels);
        return RetT::constant(*(const TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
      }
#endif
      const TT *ptr = nullptr;
      if constexpr (WithinTile) {
        ptr = (const TT *)_vector + (chn * lane_width + i);
#if ZS_ENABLE_OFB_ACCESS_CHECK
        if (i >= lane_width) {
          printf("tilevector [%s] ofb! in-tile accessing ele [%lld] out of [0, %lld)\n",
                 _nameTag.asChars(), (long long)i, (long long)lane_width);
          return RetT::constant(*(const TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
        }
#endif
      } else {
        ptr = (const TT *)_vector
              + ((i / lane_width * _numChannels + chn) * lane_width + (i % lane_width));
#if ZS_ENABLE_OFB_ACCESS_CHECK
        /// @brief check vector size overflow
        if (i >= _vectorSize) {
          printf("tilevector [%s] ofb! global accessing ele [%lld] out of [0, %lld)\n",
                 _nameTag.asChars(), (long long)i, (long long)_vectorSize);
          return RetT::constant(*(const TT *)(limits<std::uintptr_t>::max() - sizeof(TT) + 1));
        }
#endif
      }
      for (channel_counter_type d = 0; d != RetT::extent; ++d, ptr += lane_width) ret.val(d) = *ptr;
      return ret;
    }
    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr auto pack(channel_counter_type chn, const size_type i,
                        wrapt<TT> = {}) const noexcept {
      return pack(dim_c<Ns...>, chn, i, wrapt<TT>{});
    }

    template <channel_counter_type N, typename VT = value_type>
    constexpr auto array(channel_counter_type chn, const size_type i) const noexcept {
      using RetT = std::array<VT, (std::size_t)N>;
      RetT ret{};
      auto ptr = _vector + ((i / lane_width * _numChannels + chn) * lane_width + (i % lane_width));
      for (channel_counter_type d = 0; d != N; ++d, ptr += lane_width) ret[d] = *ptr;
      return ret;
    }
    /// tuple
    template <std::size_t... Is, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr auto tuple_impl(const channel_counter_type chnOffset, const size_type i,
                              index_seq<Is...>, wrapt<TT>) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if ((TT *)_vector == nullptr) {
        /// @note TBD : insert type reflection info here
        printf("tilevector [%s] tieing reinterpret_cast failed!\n", _nameTag.asChars());
        return zs::tie(*(const TT *)(Is, limits<std::uintptr_t>::max() - sizeof(TT) + 1)...);
      }
      if (chn + d > _numChannels) {
        printf("tilevector [%s] ofb! tieing chn [%d, %d) out of [0, %d)\n", _nameTag.asChars(),
               (int)chn, (int)(chn + d), (int)_numChannels);
        return zs::tie(*(const TT *)(Is, limits<std::uintptr_t>::max() - sizeof(TT) + 1)...);
      }
      if constexpr (WithinTile) {
        if (i >= lane_width) {
          printf("tilevector [%s] ofb! in-tile accessing ele [%lld] out of [0, %lld)\n",
                 _nameTag.asChars(), (long long)i, (long long)lane_width);
          return zs::tie(*(const TT *)(Is, limits<std::uintptr_t>::max() - sizeof(TT) + 1)...);
        }
      } else {
        if (i >= _vectorSize) {
          printf("tilevector [%s] ofb! global tieing ele [%lld] out of [0, %lld)\n",
                 _nameTag.asChars(), (long long)i, (long long)_vectorSize);
          return zs::tie(*(const TT *)(Is, limits<std::uintptr_t>::max() - sizeof(TT) + 1)...);
        }
      }
#endif
      if constexpr (WithinTile)
        return zs::tie(*((TT *)_vector
                         + ((size_type)chnOffset + (size_type)Is) * (size_type)lane_width + i)...);
      else {
        size_type a{}, b{};
        a = i / lane_width * _numChannels;
        b = i % lane_width;
        return zs::tie(*((TT *)_vector
                         + (a + ((size_type)chnOffset + (size_type)Is)) * (size_type)lane_width
                         + b)...);
      }
    }

    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr auto tuple(value_seq<Ns...>, const channel_counter_type chn, const size_type i,
                         wrapt<TT> = {}) const noexcept {
      return tuple_impl(chn, i, std::make_index_sequence<(Ns * ...)>{}, wrapt<TT>{});
    }
    template <auto d, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr auto tuple(const channel_counter_type chn, const size_type i,
                         wrapt<TT> = {}) const noexcept {
      return tuple(dim_c<d>, chn, i, wrapt<TT>{});
    }

    constexpr size_type size() const noexcept {
      if constexpr (WithinTile)
        return lane_width;
      else
        return _vectorSize;
    }
    constexpr channel_counter_type numChannels() const noexcept { return _numChannels; }

    conditional_t<is_const_structure, const_pointer, pointer> _vector{nullptr};
    size_type _vectorSize{0};
    channel_counter_type _numChannels{0};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    SmallString _nameTag{};
#endif
  };

  template <execspace_e ExecSpace, typename T, std::size_t Length, typename Allocator>
  constexpr decltype(auto) proxy(const TileVector<T, Length, Allocator> &vec) {
    return TileVectorUnnamedView<ExecSpace, const TileVector<T, Length, Allocator>, false>{vec};
  }
  template <execspace_e ExecSpace, typename T, std::size_t Length, typename Allocator>
  constexpr decltype(auto) proxy(TileVector<T, Length, Allocator> &vec) {
    return TileVectorUnnamedView<ExecSpace, TileVector<T, Length, Allocator>, false>{vec};
  }

  template <execspace_e ExecSpace, typename T, std::size_t Length, typename Allocator>
  constexpr decltype(auto) proxy(const TileVector<T, Length, Allocator> &vec,
                                 const SmallString &tagName) {
    auto ret = TileVectorUnnamedView<ExecSpace, const TileVector<T, Length, Allocator>, false>{vec};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ret._nameTag = tagName;
#endif
    return ret;
  }
  template <execspace_e ExecSpace, typename T, std::size_t Length, typename Allocator>
  constexpr decltype(auto) proxy(TileVector<T, Length, Allocator> &vec,
                                 const SmallString &tagName) {
    auto ret = TileVectorUnnamedView<ExecSpace, TileVector<T, Length, Allocator>, false>{vec};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ret._nameTag = tagName;
#endif
    return ret;
  }

  template <execspace_e Space, typename TileVectorT, bool WithinTile, typename = void>
  struct TileVectorView : TileVectorUnnamedView<Space, TileVectorT, WithinTile> {
    using base_t = TileVectorUnnamedView<Space, TileVectorT, WithinTile>;
#if ZS_ENABLE_OFB_ACCESS_CHECK
    using base_t::_nameTag;
#endif

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
    using base_t::_numChannels;
    using base_t::_vectorSize;
    using whole_view_type = TileVectorView<Space, TileVectorT, false>;
    using tile_view_type = TileVectorView<Space, TileVectorT, true>;
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
    using base_t::pack;
    using base_t::tuple;
    ///
    /// have to make sure that char type (might be channel_counter_type) not fit into this overload
    ///
    template <bool V = is_const_structure, typename TT = value_type,
              enable_if_all<!V, sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr std::add_lvalue_reference_t<TT> operator()(const SmallString &propName,
                                                         const channel_counter_type chn,
                                                         const size_type i,
                                                         wrapt<TT> = {}) noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (!hasProperty(propName)) {
        printf(
            "tilevector [%s] ofb! (operator()) accessing prop [%s] which is not among %d props (%d "
            "chns, %lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_numChannels,
            (long long int)_vectorSize);
        return static_cast<base_t &>(*this)(_numChannels, i, wrapt<TT>{});
      }
#endif
      return static_cast<base_t &>(*this)(_tagOffsets[propertyIndex(propName)] + chn, i,
                                          wrapt<TT>{});
    }
    template <typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr TT operator()(const SmallString &propName, const channel_counter_type chn,
                            const size_type i, wrapt<TT> = {}) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (!hasProperty(propName)) {
        printf(
            "tilevector [%s] ofb! (operator()) const accessing prop [%s] which is not among %d "
            "props (%d chns, %lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_numChannels,
            (long long int)_vectorSize);
        return static_cast<const base_t &>(*this)(_numChannels, i, wrapt<TT>{});
      }
#endif
      return static_cast<const base_t &>(*this)(_tagOffsets[propertyIndex(propName)] + chn, i,
                                                wrapt<TT>{});
    }
    template <bool V = is_const_structure, typename TT = value_type,
              enable_if_all<!V, sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr std::add_lvalue_reference_t<TT> operator()(const SmallString &propName,
                                                         const size_type i,
                                                         wrapt<TT> = {}) noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (!hasProperty(propName)) {
        printf(
            "tilevector [%s] ofb! (operator()) accessing prop [%s] which is not among %d props (%d "
            "chns, %lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_numChannels,
            (long long int)_vectorSize);
        return static_cast<base_t &>(*this)(_numChannels, i, wrapt<TT>{});
      }
#endif
      return static_cast<base_t &>(*this)(_tagOffsets[propertyIndex(propName)], i, wrapt<TT>{});
    }
    template <typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr TT operator()(const SmallString &propName, const size_type i,
                            wrapt<TT> = {}) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (!hasProperty(propName)) {
        printf(
            "tilevector [%s] ofb! (operator()) const accessing prop [%s] which is not among %d "
            "props (%d chns, %lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_numChannels,
            (long long int)_vectorSize);
        return static_cast<const base_t &>(*this)(_numChannels, i, wrapt<TT>{});
      }
#endif
      return static_cast<const base_t &>(*this)(_tagOffsets[propertyIndex(propName)], i,
                                                wrapt<TT>{});
    }
    template <bool V = is_const_structure, bool InTile = WithinTile, enable_if_all<!V, !InTile> = 0>
    constexpr auto tile(const size_type tileid) noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (tileid >= (_vectorSize + lane_width - 1) / lane_width) {
        printf("tilevector [%s] ofb! global accessing tile %d out of %d blocks\n",
               _nameTag.asChars(), (int)tileid, (int)(_vectorSize + lane_width - 1) / lane_width);
      }
#endif
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
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (tileid >= (_vectorSize + lane_width - 1) / lane_width) {
        printf("tilevector [%s] ofb! const global accessing tile %d out of %d blocks\n",
               _nameTag.asChars(), (int)tileid, (int)(_vectorSize + lane_width - 1) / lane_width);
      }
#endif
      return TileVectorView<Space, const_tile_vector_type, true>{
          this->_vector + tileid * lane_width * this->_numChannels,
          lane_width,
          this->_numChannels,
          _tagNames,
          _tagOffsets,
          _tagSizes,
          _N};
    }

    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr auto pack(value_seq<Ns...>, const SmallString &propName, const size_type i,
                        wrapt<TT> = {}) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (!hasProperty(propName)) {
        printf(
            "tilevector [%s] ofb! (pack) accessing prop [%s] which is not among %d props (%d chns, "
            "%lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_numChannels,
            (long long int)_vectorSize);
        // return static_cast<const base_t &>(*this).pack(dim_c<Ns...>, _numChannels, i,
        // wrapt<TT>{});
        using RetT = decltype(static_cast<const base_t &>(*this).pack(
            dim_c<Ns...>, _tagOffsets[propertyIndex(propName)], i, wrapt<TT>{}));
        return RetT::zeros();
      }
#endif
      return static_cast<const base_t &>(*this).pack(
          dim_c<Ns...>, _tagOffsets[propertyIndex(propName)], i, wrapt<TT>{});
    }
    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr auto pack(const SmallString &propName, const size_type i,
                        wrapt<TT> = {}) const noexcept {
      return pack(dim_c<Ns...>, propName, i, wrapt<TT>{});
    }

    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr auto pack(value_seq<Ns...>, const SmallString &propName,
                        const channel_counter_type chn, const size_type i,
                        wrapt<TT> = {}) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (!hasProperty(propName)) {
        printf(
            "tilevector [%s] ofb! (pack) accessing prop [%s] which is not among %d props (%d chns, "
            "%lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_numChannels,
            (long long int)_vectorSize);
        // return static_cast<const base_t &>(*this).pack(dim_c<Ns...>, _numChannels, i,
        // wrapt<TT>{});
        using RetT = decltype(static_cast<const base_t &>(*this).pack(
            dim_c<Ns...>, _tagOffsets[propertyIndex(propName)] + chn, i, wrapt<TT>{}));
        return RetT::zeros();
      }
#endif
      return static_cast<const base_t &>(*this).pack(
          dim_c<Ns...>, _tagOffsets[propertyIndex(propName)] + chn, i, wrapt<TT>{});
    }
    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr auto pack(const SmallString &propName, const channel_counter_type chn,
                        const size_type i, wrapt<TT> = {}) const noexcept {
      return pack(dim_c<Ns...>, propName, chn, i, wrapt<TT>{});
    }

    template <channel_counter_type N, typename VT = value_type>
    constexpr auto array(const SmallString &propName, const size_type i) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (!hasProperty(propName)) {
        printf(
            "tilevector [%s] ofb! (array) accessing prop [%s] which is not among %d props (%d "
            "chns, %lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_numChannels,
            (long long int)_vectorSize);
        return static_cast<const base_t &>(*this).template array<N, VT>(_numChannels, i);
      }
#endif
      return static_cast<const base_t &>(*this).template array<N, VT>(
          _tagOffsets[propertyIndex(propName)], i);
    }

    /// @brief tieing
    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr auto tuple(value_seq<Ns...>, const SmallString &propName, const size_type i,
                         wrapt<TT> = {}) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (!hasProperty(propName)) {
        printf(
            "tilevector [%s] ofb! (tuple) const accessing prop [%s] which is not among %d props "
            "(%d chns, %lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_numChannels,
            (long long int)_vectorSize);
        return static_cast<const base_t &>(*this).tuple(dim_c<Ns...>, _numChannels, i, wrapt<TT>{});
      }
#endif
      return static_cast<const base_t &>(*this).tuple(
          dim_c<Ns...>, _tagOffsets[propertyIndex(propName)], i, wrapt<TT>{});
    }
    template <auto d, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr auto tuple(const SmallString &propName, const size_type i,
                         wrapt<TT> = {}) const noexcept {
      return tuple(dim_c<d>, propName, i, wrapt<TT>{});
    }

    template <auto... Ns, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr auto tuple(value_seq<Ns...>, const SmallString &propName,
                         const channel_counter_type chn, const size_type i,
                         wrapt<TT> = {}) const noexcept {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (!hasProperty(propName)) {
        printf(
            "tilevector [%s] ofb! (tuple) const accessing prop [%s] which is not among %d props "
            "(%d chns, %lld eles) in total\n",
            _nameTag.asChars(), propName.asChars(), (int)_N, (int)_numChannels,
            (long long int)_vectorSize);
        return static_cast<const base_t &>(*this).tuple(dim_c<Ns...>, _numChannels, i, wrapt<TT>{});
      }
#endif
      return static_cast<const base_t &>(*this).tuple(
          dim_c<Ns...>, _tagOffsets[propertyIndex(propName)] + chn, i, wrapt<TT>{});
    }
    template <auto d, typename TT = value_type,
              enable_if_all<sizeof(TT) == sizeof(value_type), is_same_v<TT, remove_cvref_t<TT>>,
                            (std::alignment_of_v<TT> == std::alignment_of_v<value_type>)> = 0>
    constexpr auto tuple(const SmallString &propName, const channel_counter_type chn,
                         const size_type i, wrapt<TT> = {}) const noexcept {
      return tuple(dim_c<d>, propName, chn, i, wrapt<TT>{});
    }

    const SmallString *_tagNames{nullptr};
    const channel_counter_type *_tagOffsets{nullptr};
    const channel_counter_type *_tagSizes{nullptr};
    channel_counter_type _N{0};
  };

  template <execspace_e ExecSpace, typename T, std::size_t Length, typename Allocator>
  decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                       const TileVector<T, Length, Allocator> &vec) {
    for (auto &&tag : tagNames)
      if (!vec.hasProperty(tag))
        throw std::runtime_error(
            fmt::format("tilevector attribute [\"{}\"] not exists", (std::string)tag));
    return TileVectorView<ExecSpace, const TileVector<T, Length, Allocator>, false>{tagNames, vec};
  }
  template <execspace_e ExecSpace, typename T, std::size_t Length, typename Allocator>
  decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                       TileVector<T, Length, Allocator> &vec) {
    for (auto &&tag : tagNames)
      if (!vec.hasProperty(tag))
        throw std::runtime_error(
            fmt::format("tilevector attribute [\"{}\"] not exists\n", (std::string)tag));
    return TileVectorView<ExecSpace, TileVector<T, Length, Allocator>, false>{tagNames, vec};
  }

  /// tagged tilevector for debug
  template <execspace_e ExecSpace, typename T, std::size_t Length, typename Allocator>
  decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                       const TileVector<T, Length, Allocator> &vec, const SmallString &tagName) {
    auto ret = TileVectorView<ExecSpace, const TileVector<T, Length, Allocator>, false>{{}, vec};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ret._nameTag = tagName;
#endif
    return ret;
  }
  template <execspace_e ExecSpace, typename T, std::size_t Length, typename Allocator>
  decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                       TileVector<T, Length, Allocator> &vec, const SmallString &tagName) {
    auto ret = TileVectorView<ExecSpace, TileVector<T, Length, Allocator>, false>{{}, vec};
#if ZS_ENABLE_OFB_ACCESS_CHECK
    ret._nameTag = tagName;
#endif
    return ret;
  }

}  // namespace zs