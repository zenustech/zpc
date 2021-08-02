#pragma once
#include "Vector.hpp"
#include "zensim/math/Vec.h"
#include "zensim/memory/Allocator.h"
#include "zensim/resource/Resource.h"
#include "zensim/tpls/magic_enum/magic_enum.hpp"
#include "zensim/types/Iterator.h"
#include "zensim/types/Polymorphism.h"
#include "zensim/types/RuntimeStructurals.hpp"
#include "zensim/types/SmallVector.hpp"

namespace zs {

  template <auto Length, typename Snode, typename ChnCounter>  ///< length should be power of 2
  using tile_snode = ds::snode_t<ds::decorations<ds::soa>, ds::static_domain<Length>, tuple<Snode>,
                                 tuple<ChnCounter>>;

  template <auto Length, typename Snode, typename ChnCounter, typename Index> using aosoa_snode
      = ds::snode_t<ds::static_decorator<>, ds::uniform_domain<0, Index, 1, index_seq<0>>,
                    tuple<tile_snode<Length, Snode, ChnCounter>>, vseq_t<1>>;
  //                    typename gen_seq<sizeof...(Snodes)>::template uniform_vseq<1>

  template <auto Length, typename T, typename ChnCounter, typename Index> using aosoa_instance
      = ds::instance_t<ds::dense, aosoa_snode<Length, wrapt<T>, ChnCounter, Index>>;

  template <typename T, auto Length = 8, typename Index = std::size_t,
            typename ChnCounter = unsigned char>
  struct TileVector {
    static_assert(is_same_v<T, remove_cvref_t<T>>, "T is not cvref-unqualified type!");
    static_assert(std::is_default_constructible_v<T> && std::is_trivially_copyable_v<T>,
                  "element is not default-constructible or trivially-copyable!");

    using base_t = aosoa_instance<Length, T, ChnCounter, Index>;
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
    static constexpr auto lane_width = Length;

    static_assert(
        std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_swap::value,
        "allocator should propagate on copy, move and swap (for impl simplicity)!");

    constexpr decltype(auto) memoryLocation() noexcept { return _allocator.location; }
    constexpr decltype(auto) memoryLocation() const noexcept { return _allocator.location; }
    constexpr ProcID devid() const noexcept { return memoryLocation().devid(); }
    constexpr memsrc_e memspace() const noexcept { return memoryLocation().memspace(); }
    constexpr decltype(auto) allocator() const noexcept { return _allocator; }

    constexpr base_t &self() noexcept { return _inst; }
    constexpr const base_t &self() const noexcept { return _inst; }

    constexpr TileVector(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : _allocator{get_memory_source(mre, devid)}, _tags(0), _size{0} {
      _inst = buildInstance(0, 0);
    }
#if 0
    TileVector(channel_counter_type numChns, size_type count = 0, memsrc_e mre = memsrc_e::host,
               ProcID devid = -1)
        : _allocator{get_memory_source(mre, devid)},
          _tags(numChns),
          _size{count} {
      _inst = buildInstance(numChns, count);
    }
#endif
    TileVector(const std::vector<PropertyTag> &channelTags, size_type count = 0,
               memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : _allocator{get_memory_source(mre, devid)}, _tags{channelTags}, _size{count} {
      _inst = buildInstance(numTotalChannels(channelTags), count);
    }
    TileVector(const allocator_type &allocator, const std::vector<PropertyTag> &channelTags,
               size_type count = 0)
        : _allocator{allocator}, _tags{channelTags}, _size{count} {
      _inst = buildInstance(numTotalChannels(channelTags), count);
    }

    ~TileVector() {
      if (self().address() && self().node().extent() > 0) self().dealloc(_allocator);
    }
    void initPropertyTags(const channel_counter_type N) {
      _tagNames = Vector<SmallString>{_allocator, static_cast<std::size_t>(N)};
      _tagSizes = Vector<channel_counter_type>{_allocator, static_cast<std::size_t>(N)};
      _tagOffsets = Vector<channel_counter_type>{_allocator, static_cast<std::size_t>(N)};
    }

    static auto numTotalChannels(const std::vector<PropertyTag> &tags) {
      tuple_element_t<1, PropertyTag> cnt = 0;
      for (std::size_t i = 0; i != tags.size(); ++i) cnt += tags[i].template get<1>();
      return cnt;
    }
    struct iterator_impl : IteratorInterface<iterator_impl> {
      constexpr iterator_impl(const base_t &range, size_type idx, channel_counter_type chn = 0)
          : _base{range.data()},
            _idx{idx},
            _chn{chn},
            _numChannels{range.node().child(wrapv<0>{}).channel_count()} {}

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
      channel_counter_type _chn{0}, _numChannels{0};
    };
    struct const_iterator_impl : IteratorInterface<const_iterator_impl> {
      constexpr const_iterator_impl(const base_t &range, size_type idx,
                                    channel_counter_type chn = 0)
          : _base{range.data()},
            _idx{idx},
            _chn{chn},
            _numChannels{range.node().child(wrapv<0>{}).channel_count()} {}

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
      channel_counter_type _chn{0}, _numChannels{0};
    };
    using iterator = LegacyIterator<iterator_impl>;
    using const_iterator = LegacyIterator<const_iterator_impl>;

    constexpr auto begin() noexcept { return make_iterator<iterator_impl>(self(), 0); }
    constexpr auto end() noexcept { return make_iterator<iterator_impl>(self(), size()); }
    constexpr auto begin() const noexcept { return make_iterator<const_iterator_impl>(self(), 0); }
    constexpr auto end() const noexcept {
      return make_iterator<const_iterator_impl>(self(), size());
    }

    /// capacity
    constexpr size_type size() const noexcept { return _size; }
    constexpr size_type capacity() const noexcept { return self().node().extent(); }
    constexpr bool empty() noexcept { return size() == 0; }
    constexpr pointer head() const noexcept { return reinterpret_cast<pointer>(self().address()); }
    constexpr const_pointer data() const noexcept {
      return reinterpret_cast<const_pointer>(self().address());
    }
    constexpr pointer data() noexcept { return reinterpret_cast<pointer>(self().address()); }

    /// element access
    constexpr reference operator[](
        const std::tuple<channel_counter_type, size_type> index) noexcept {
      const auto [chn, idx] = index;
      return self()(idx / lane_width)(chn, idx % lane_width);
    }
    constexpr conditional_t<std::is_fundamental_v<value_type>, value_type, const_reference>
    operator[](const std::tuple<channel_counter_type, size_type> index) const noexcept {
      const auto [chn, idx] = index;
      return self()(idx / lane_width)(chn, idx % lane_width);
    }
    /// ctor, assignment operator
    TileVector(const TileVector &o) : _allocator{o._allocator}, _tags{o._tags}, _size{o.size()} {
      _inst = buildInstance(o.numChannels(), this->size());
      if (ds::snode_size(o.self().template node<0>()) > 0)
        copy(MemoryEntity{memoryLocation(), (void *)self().address()},
             MemoryEntity{o.memoryLocation(), (void *)o.data()},
             ds::snode_size(o.self().template node<0>()));
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
      TileVector ret{allocator, _tags, size()};
      copy(MemoryEntity{allocator.location, (void *)ret.data()},
           MemoryEntity{memoryLocation(), (void *)data()},
           ds::snode_size(self().template node<0>()));
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
      self() = std::exchange(o.self(), defaultVector.self());
      _allocator = std::move(o._allocator);
      _tags = std::move(o._tags);
      _size = std::exchange(o._size, defaultVector.size());
    }
    /// make move-assignment safe for self-assignment
    TileVector &operator=(TileVector &&o) noexcept {
      if (this == &o) return *this;
      TileVector tmp(std::move(o));
      swap(tmp);
      return *this;
    }
    void swap(TileVector &o) noexcept {
      std::swap(self(), o.self());
      std::swap(_allocator, o._allocator);
      std::swap(_tags, o._tags);
      std::swap(_size, o._size);
    }
    friend void swap(TileVector &a, TileVector &b) { a.swap(b); }

    void clear() { *this = TileVector{_allocator, _tags, 0}; }

    constexpr channel_counter_type numChannels() const noexcept {
      return self().node().child(wrapv<0>{}).channel_count();
    }
    constexpr channel_counter_type numProperties() const noexcept { return _tags.size(); }
    constexpr auto propertyOffsets() const {
      std::vector<channel_counter_type> offsets(_tags.size());
      for (Vector<PropertyTag>::size_type i = 0; i != _tags.size(); ++i)
        offsets[i] = (i ? offsets[i - 1] + _tags[i - 1].template get<1>() : 0);
      return offsets;
    }
    void preparePropertyNames(const std::vector<SmallString> &propNames) {
      const auto N = propNames.size();
      if (N <= 0) return;
      Vector<SmallString> hostPropNames{N, memsrc_e::host, -1};
      Vector<channel_counter_type> hostPropSizes{N, memsrc_e::host, -1};
      Vector<channel_counter_type> hostPropOffsets{N, memsrc_e::host, -1};

      for (std::size_t i = 0, ed = propNames.size(); i != ed; ++i) hostPropNames[i] = propNames[i];
      const auto offsets = propertyOffsets();
      for (std::size_t dst = 0; dst < N; ++dst) {
        Vector<PropertyTag>::size_type i = 0;
        for (; i < _tags.size(); ++i) {
          if (_tags[i].template get<0>() == propNames[dst]) {
            hostPropSizes[dst] = _tags[i].template get<1>();
            hostPropOffsets[dst] = offsets[i];
            break;
          }
        }
        if (i == _tags.size()) {
          hostPropSizes[dst] = 0;
          hostPropOffsets[dst] = -1;
        }
      }
      initPropertyTags(N);
      copy(MemoryEntity{memoryLocation(), (void *)_tagNames.data()},
           MemoryEntity{hostPropNames.memoryLocation(), (void *)hostPropNames.data()},
           sizeof(SmallString) * N);
      copy(MemoryEntity{memoryLocation(), (void *)_tagSizes.data()},
           MemoryEntity{hostPropSizes.memoryLocation(), (void *)hostPropSizes.data()},
           sizeof(channel_counter_type) * N);
      copy(MemoryEntity{memoryLocation(), (void *)_tagOffsets.data()},
           MemoryEntity{hostPropOffsets.memoryLocation(), (void *)hostPropOffsets.data()},
           sizeof(channel_counter_type) * N);
    }

    constexpr const SmallString *tagNameHandle() const noexcept { return _tagNames.data(); }
    constexpr const channel_counter_type *tagSizeHandle() const noexcept {
      return _tagSizes.data();
    }
    constexpr const channel_counter_type *tagOffsetHandle() const noexcept {
      return _tagOffsets.data();
    }

  protected:
    constexpr auto buildInstance(channel_counter_type numChns, size_type capacity) {
      using namespace ds;
      tile_snode<lane_width, wrapt<T>, channel_counter_type> tilenode{
          ds::decorations<ds::soa>{}, static_domain<lane_width>{}, zs::make_tuple(wrapt<T>{}),
          zs::make_tuple(numChns)};
      uniform_domain<0, size_type, 1, index_seq<0>> dom{wrapv<0>{},
                                                        (capacity + lane_width - 1) / lane_width};
      aosoa_snode<lane_width, wrapt<T>, channel_counter_type, size_type> node{
          ds::static_decorator<>{}, dom, zs::make_tuple(tilenode), vseq_t<1>{}};
      auto inst = instance{wrapv<dense>{}, zs::make_tuple(node)};
      if (capacity) inst.alloc(_allocator);
      return inst;
    }

    base_t _inst;
    allocator_type _allocator;
    std::vector<PropertyTag> _tags;  // on host
    /// for proxy use
    Vector<SmallString> _tagNames;
    Vector<channel_counter_type> _tagSizes;
    Vector<channel_counter_type> _tagOffsets;
    size_type _size{0};  // element size
  };

  template <execspace_e, typename TileVectorT, typename = void> struct TileVectorView;
  template <execspace_e, typename TileVectorT, typename = void> struct TileVectorUnnamedView;

  template <execspace_e Space, typename TileVectorT>
  struct TileVectorUnnamedView<Space, TileVectorT> {
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
    constexpr reference operator()(channel_counter_type chn, const size_type i) {
      return *(_vector + (i / lane_width * _numChannels + chn) * lane_width + i % lane_width);
    }
    constexpr const_reference operator()(channel_counter_type chn, const size_type i) const {
      return *(_vector + (i / lane_width * _numChannels + chn) * lane_width + i % lane_width);
    }

    constexpr reference val(channel_counter_type chn, const size_type i) {
      return *(_vector + (i / lane_width * _numChannels + chn) * lane_width + i % lane_width);
    }
    constexpr const_reference val(channel_counter_type chn, const size_type i) const {
      return *(_vector + (i / lane_width * _numChannels + chn) * lane_width + i % lane_width);
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

  template <execspace_e Space, typename TileVectorT>
  struct TileVectorUnnamedView<Space, const TileVectorT> {
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
    constexpr const_reference operator()(channel_counter_type chn, const size_type i) const {
      return *(_vector + (i / lane_width * _numChannels + chn) * lane_width + i % lane_width);
    }
    constexpr const_reference val(channel_counter_type chn, const size_type i) const {
      return *(_vector + (i / lane_width * _numChannels + chn) * lane_width + i % lane_width);
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
  template <execspace_e ExecSpace, typename T, auto Length, typename IndexT, typename ChnT>
  constexpr decltype(auto) proxy(const TileVector<T, Length, IndexT, ChnT> &vec) {
    return TileVectorUnnamedView<ExecSpace, const TileVector<T, Length, IndexT, ChnT>>{vec};
  }
  template <execspace_e ExecSpace, typename T, auto Length, typename IndexT, typename ChnT>
  constexpr decltype(auto) proxy(TileVector<T, Length, IndexT, ChnT> &vec) {
    return TileVectorUnnamedView<ExecSpace, TileVector<T, Length, IndexT, ChnT>>{vec};
  }

  template <execspace_e Space, typename TileVectorT> struct TileVectorView<Space, TileVectorT>
      : TileVectorUnnamedView<Space, TileVectorT> {
    using base_t = TileVectorUnnamedView<Space, TileVectorT>;
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
        : base_t{tilevector}, N{static_cast<channel_counter_type>(tagNames.size())} {
      tilevector.preparePropertyNames(tagNames);
      _tagNames = tilevector.tagNameHandle();
      _tagOffsets = tilevector.tagOffsetHandle();
      _tagSizes = tilevector.tagSizeHandle();
    }

    constexpr auto propIndex(const char propName[]) const {
      channel_counter_type i = 0;
      for (; i < N; ++i)
        if (_tagNames[i] == propName) break;
      return i;
    }
    constexpr bool hasProp(const char propName[]) const { return propIndex(propName) != N; }

    constexpr reference operator()(const char propName[], const channel_counter_type chn,
                                   const size_type i) {
      return static_cast<base_t &>(*this)(_tagOffsets[propIndex(propName)] + chn, i);
    }
    constexpr const_reference operator()(const char propName[], const channel_counter_type chn,
                                         const size_type i) const {
      return static_cast<const base_t &>(*this)(_tagOffsets[propIndex(propName)] + chn, i);
    }
    constexpr reference val(const char propName[], const size_type i) {
      return static_cast<base_t &>(*this)(_tagOffsets[propIndex(propName)], i);
    }
    constexpr const_reference val(const char propName[], const size_type i) const {
      return static_cast<const base_t &>(*this)(_tagOffsets[propIndex(propName)], i);
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
    channel_counter_type N{0};
  };
  template <execspace_e Space, typename TileVectorT> struct TileVectorView<Space, const TileVectorT>
      : TileVectorUnnamedView<Space, const TileVectorT> {
    using base_t = TileVectorUnnamedView<Space, const TileVectorT>;
    using const_pointer = typename TileVectorT::const_pointer;
    using value_type = typename TileVectorT::value_type;
    using const_reference = typename TileVectorT::const_reference;
    using size_type = typename TileVectorT::size_type;
    using channel_counter_type = typename TileVectorT::channel_counter_type;
    static constexpr auto lane_width = TileVectorT::lane_width;

    constexpr TileVectorView() = default;
    ~TileVectorView() = default;
    explicit constexpr TileVectorView(const std::vector<SmallString> &tagNames,
                                      TileVectorT &tilevector)
        : base_t{tilevector}, N{static_cast<channel_counter_type>(tagNames.size())} {
      tilevector.preparePropertyNames(tagNames);
      _tagNames = tilevector.tagNameHandle();
      _tagOffsets = tilevector.tagOffsetHandle();
      _tagSizes = tilevector.tagSizeHandle();
    }

    constexpr auto propIndex(const char propName[]) const {
      channel_counter_type i = 0;
      for (; i < N; ++i)
        if (_tagNames[i] == propName) break;
      return i;
    }
    constexpr bool hasProp(const char propName[]) const { return propIndex(propName) != N; }

    constexpr const_reference operator()(const char propName[], const channel_counter_type chn,
                                         const size_type i) const {
      return static_cast<const base_t &>(*this)(_tagOffsets[propIndex(propName)] + chn, i);
    }
    constexpr const_reference val(const char propName[], const size_type i) const {
      return static_cast<const base_t &>(*this)(_tagOffsets[propIndex(propName)], i);
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
    channel_counter_type N{0};
  };

  template <execspace_e ExecSpace, typename T, auto Length, typename IndexT, typename ChnT>
  constexpr decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                                 const TileVector<T, Length, IndexT, ChnT> &vec) {
    return TileVectorView<ExecSpace, const TileVector<T, Length, IndexT, ChnT>>{tagNames, vec};
  }
  template <execspace_e ExecSpace, typename T, auto Length, typename IndexT, typename ChnT>
  constexpr decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                                 TileVector<T, Length, IndexT, ChnT> &vec) {
    return TileVectorView<ExecSpace, TileVector<T, Length, IndexT, ChnT>>{tagNames, vec};
  }

}  // namespace zs