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
  struct TileVector : MemoryHandle {
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

    constexpr MemoryHandle &base() noexcept { return static_cast<MemoryHandle &>(*this); }
    constexpr const MemoryHandle &base() const noexcept {
      return static_cast<const MemoryHandle &>(*this);
    }
    constexpr base_t &self() noexcept { return _inst; }
    constexpr const base_t &self() const noexcept { return _inst; }

    constexpr TileVector(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : MemoryHandle{mre, devid}, _allocator{get_memory_source(mre, devid)}, _tags(0), _size{0} {
      _inst = buildInstance(mre, devid, 0, 0);
    }
    TileVector(channel_counter_type numChns, size_type count = 0, memsrc_e mre = memsrc_e::host,
               ProcID devid = -1)
        : MemoryHandle{mre, devid},
          _allocator{get_memory_source(mre, devid)},
          _tags(numChns),
          _size{count} {
      _inst = buildInstance(mre, devid, numChns, count);
    }
    TileVector(const std::vector<PropertyTag> &channelTags, size_type count = 0,
               memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : MemoryHandle{mre, devid},
          _allocator{get_memory_source(mre, devid)},
          _tags{channelTags},
          _size{count} {
      _inst = buildInstance(mre, devid, numTotalChannels(channelTags), count);
    }
    TileVector(mr_t *mr, const std::vector<PropertyTag> &channelTags, size_type count = 0,
               memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : MemoryHandle{mre, devid}, _allocator{mr}, _tags{channelTags}, _size{count} {
      _inst = buildInstance(mre, devid, numTotalChannels(channelTags), count);
    }
    TileVector(const SharedHolder<mr_t> &mr, const std::vector<PropertyTag> &channelTags,
               size_type count = 0, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : MemoryHandle{mre, devid}, _allocator{mr}, _tags{channelTags}, _size{count} {
      _inst = buildInstance(mre, devid, numTotalChannels(channelTags), count);
    }
    TileVector(const allocator_type &allocator, const std::vector<PropertyTag> &channelTags,
               size_type count = 0, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : MemoryHandle{mre, devid}, _allocator{allocator}, _tags{channelTags}, _size{count} {
      _inst = buildInstance(mre, devid, numTotalChannels(channelTags), count);
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
          : _range{range}, _idx{idx}, _chn{chn} {}

      constexpr reference dereference() { return _range(_idx / lane_width)(_idx % lane_width); }
      constexpr bool equal_to(iterator_impl it) const noexcept { return it._idx == _idx; }
      constexpr void advance(difference_type offset) noexcept { _idx += offset; }
      constexpr difference_type distance_to(iterator_impl it) const noexcept {
        return it._idx - _idx;
      }

    protected:
      base_t _range{};
      size_type _idx{0};
      channel_counter_type _chn{};
    };
    struct const_iterator_impl : IteratorInterface<const_iterator_impl> {
      constexpr const_iterator_impl(const base_t &range, size_type idx,
                                    channel_counter_type chn = 0)
          : _range{range}, _idx{idx}, _chn{chn} {}

      constexpr const_reference dereference() {
        return _range(_idx / lane_width)(_idx % lane_width);
      }
      constexpr bool equal_to(const_iterator_impl it) const noexcept { return it._idx == _idx; }
      constexpr void advance(difference_type offset) noexcept { _idx += offset; }
      constexpr difference_type distance_to(const_iterator_impl it) const noexcept {
        return it._idx - _idx;
      }

    protected:
      base_t _range{};
      size_type _idx{0};
      channel_counter_type _chn{};
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
    TileVector(const TileVector &o)
        : MemoryHandle{o.base()}, _allocator{o._allocator}, _tags{o._tags}, _size{o.size()} {
      _inst = buildInstance(this->memspace(), this->devid(), o.numChannels(), this->size());
      if (ds::snode_size(o.self().template node<0>()) > 0)
        copy(MemoryEntity{base(), (void *)self().address()},
             MemoryEntity{o.base(), (void *)o.data()}, ds::snode_size(o.self().template node<0>()));
    }
    TileVector &operator=(const TileVector &o) {
      if (this == &o) return *this;
      TileVector tmp(o);
      swap(tmp);
      return *this;
    }
    TileVector clone(const MemoryHandle &mh, const allocator_type &allocator) const {
      // capacity() is the count of tiles
      // use size() that represents the number of elements!
      TileVector ret{allocator, _tags, size(), mh.memspace(), mh.devid()};
      copy(MemoryEntity{mh, (void *)ret.data()}, MemoryEntity{base(), (void *)data()},
           ds::snode_size(self().template node<0>()));
      return ret;
    }
    TileVector clone(const MemoryHandle &mh) const {
      return clone(mh, get_memory_source(mh.memspace(), mh.devid()));
    }
    /// assignment or destruction after std::move
    /// https://www.youtube.com/watch?v=ZG59Bqo7qX4
    /// explicit noexcept
    /// leave the source object in a valid (default constructed) state
    TileVector(TileVector &&o) noexcept {
      const TileVector defaultVector{};
      base() = std::exchange(o.base(), defaultVector.base());
      self() = std::exchange(o.self(), defaultVector.self());
      _allocator = std::exchange(o._allocator, defaultVector._allocator);
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
      std::swap(base(), o.base());
      std::swap(self(), o.self());
      std::swap(_allocator, o._allocator);
      std::swap(_tags, o._tags);
      std::swap(_size, o._size);
    }

    void clear() { *this = TileVector{_allocator, _tags, 0, memspace(), devid()}; }

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
      copy(MemoryEntity{base(), (void *)_tagNames.data()},
           MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)hostPropNames.data()},
           sizeof(SmallString) * N);
      copy(MemoryEntity{base(), (void *)_tagSizes.data()},
           MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)hostPropSizes.data()},
           sizeof(channel_counter_type) * N);
      copy(MemoryEntity{base(), (void *)_tagOffsets.data()},
           MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)hostPropOffsets.data()},
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
    constexpr auto buildInstance(memsrc_e mre, ProcID devid, channel_counter_type numChns,
                                 size_type capacity) {
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
  struct TileVectorUnnamedView<Space, TileVectorT> : TileVectorT::base_t {
    using value_type = typename TileVectorT::value_type;
    using tile_vector_t = typename TileVectorT::base_t;
    using size_type = typename TileVectorT::size_type;
    using channel_counter_type = typename TileVectorT::channel_counter_type;
    static constexpr auto lane_width = TileVectorT::lane_width;

    constexpr TileVectorUnnamedView() = default;
    ~TileVectorUnnamedView() = default;
    explicit constexpr TileVectorUnnamedView(TileVectorT &tilevector)
        : tile_vector_t{tilevector.self()}, _vectorSize{tilevector.size()} {}
    constexpr auto &operator()(const channel_counter_type c, const size_type i) {
      return static_cast<tile_vector_t &>(*this)(i / lane_width)(c, i % lane_width);
    }
    constexpr const auto &operator()(const channel_counter_type c, const size_type i) const {
      return static_cast<const tile_vector_t &>(*this)(i / lane_width)(c, i % lane_width);
    }

    constexpr auto &val(channel_counter_type chn, const size_type i) {
      return static_cast<tile_vector_t &>(*this)(i / lane_width)(chn, i % lane_width);
    }
    constexpr const auto &val(channel_counter_type chn, const size_type i) const {
      return static_cast<const tile_vector_t &>(*this)(i / lane_width)(chn, i % lane_width);
    }

    template <auto... Ns> constexpr auto pack(channel_counter_type chn, const size_type i) const {
      using RetT = vec<value_type, Ns...>;
      RetT ret{};
      const auto a = i / lane_width, b = i % lane_width;
      for (channel_counter_type d = 0; d < RetT::extent; ++d)
        ret.val(d) = static_cast<const tile_vector_t &>(*this)(a)(chn + d, b);
      return ret;
    }
    template <std::size_t... Is> constexpr auto tuple_impl(const channel_counter_type chnOffset,
                                                           const size_type i, index_seq<Is...>) {
      const auto a = i / lane_width, b = i % lane_width;
      return zs::forward_as_tuple(static_cast<tile_vector_t &>(*this)(a)(chnOffset + Is, b)...);
    }
    template <std::size_t... Is> constexpr auto stdtuple_impl(const channel_counter_type chnOffset,
                                                              const size_type i, index_seq<Is...>) {
      const auto a = i / lane_width, b = i % lane_width;
      return std::forward_as_tuple(static_cast<tile_vector_t &>(*this)(a)(chnOffset + Is, b)...);
    }
    template <auto d> constexpr auto tuple(channel_counter_type chn, const size_type i) {
      return tuple_impl(chn, i, std::make_index_sequence<d>{});
    }
    template <auto d> constexpr auto stdtuple(channel_counter_type chn, const size_type i) {
      return stdtuple_impl(chn, i, std::make_index_sequence<d>{});
    }

    constexpr channel_counter_type numChannels() const noexcept {
      return this->node().child(wrapv<0>{}).channel_count();
    }

    size_type _vectorSize{0};
  };

  template <execspace_e Space, typename TileVectorT> struct TileVectorView<Space, TileVectorT>
      : TileVectorT::base_t {
    using value_type = typename TileVectorT::value_type;
    using tile_vector_t = typename TileVectorT::base_t;
    using size_type = typename TileVectorT::size_type;
    using channel_counter_type = typename TileVectorT::channel_counter_type;
    static constexpr auto lane_width = TileVectorT::lane_width;

    constexpr TileVectorView() = default;
    ~TileVectorView() = default;
    explicit constexpr TileVectorView(const std::vector<SmallString> &tagNames,
                                      TileVectorT &tilevector)
        : tile_vector_t{tilevector.self()},
          _vectorSize{tilevector.size()},
          N{static_cast<channel_counter_type>(tagNames.size())} {
      tilevector.preparePropertyNames(tagNames);
      _tagNames = tilevector.tagNameHandle();
      _tagOffsets = tilevector.tagOffsetHandle();
      _tagSizes = tilevector.tagSizeHandle();
    }
    constexpr auto &operator()(const channel_counter_type c, const size_type i) {
      return static_cast<tile_vector_t &>(*this)(i / lane_width)(c, i % lane_width);
    }
    constexpr const auto &operator()(const channel_counter_type c, const size_type i) const {
      return static_cast<const tile_vector_t &>(*this)(i / lane_width)(c, i % lane_width);
    }
    constexpr auto propIndex(const char propName[]) const {
      channel_counter_type i = 0;
      for (; i < N; ++i)
        if (_tagNames[i] == propName) break;
      return i;
    }
    constexpr bool hasProp(const char propName[]) const { return propIndex(propName) != N; }

    constexpr auto &operator()(const char propName[], const channel_counter_type c,
                               const size_type i) {
      return static_cast<tile_vector_t &>(*this)(i / lane_width)(
          _tagOffsets[propIndex(propName)] + c, i % lane_width);
    }
    constexpr const auto &operator()(const char propName[], const channel_counter_type c,
                                     const size_type i) const {
      return static_cast<const tile_vector_t &>(*this)(i / lane_width)(
          _tagOffsets[propIndex(propName)] + c, i % lane_width);
    }
    constexpr auto &val(const char propName[], const size_type i) {
      return static_cast<tile_vector_t &>(*this)(i / lane_width)(_tagOffsets[propIndex(propName)],
                                                                 i % lane_width);
    }
    constexpr const auto &val(const char propName[], const size_type i) const {
      return static_cast<const tile_vector_t &>(*this)(i / lane_width)(
          _tagOffsets[propIndex(propName)], i % lane_width);
    }

    template <auto... Ns> constexpr auto pack(const char propName[], const size_type i) const {
      using RetT = vec<value_type, Ns...>;
      RetT ret{};
      const auto a = i / lane_width, b = i % lane_width;
      const auto chnOffset = _tagOffsets[propIndex(propName)];
      for (channel_counter_type d = 0; d < RetT::extent; ++d)
        ret.val(d) = static_cast<const tile_vector_t &>(*this)(a)(chnOffset + d, b);
      return ret;
    }
    template <std::size_t... Is> constexpr auto tuple_impl(const channel_counter_type chnOffset,
                                                           const size_type i, index_seq<Is...>) {
      const auto a = i / lane_width, b = i % lane_width;
      return zs::forward_as_tuple(static_cast<tile_vector_t &>(*this)(a)(chnOffset + Is, b)...);
    }
    template <std::size_t... Is> constexpr auto stdtuple_impl(const channel_counter_type chnOffset,
                                                              const size_type i, index_seq<Is...>) {
      const auto a = i / lane_width, b = i % lane_width;
      return std::forward_as_tuple(static_cast<tile_vector_t &>(*this)(a)(chnOffset + Is, b)...);
    }
    template <auto d> constexpr auto tuple(const char propName[], const size_type i) {
      return tuple_impl(_tagOffsets[propIndex(propName)], i, std::make_index_sequence<d>{});
    }
    template <auto d> constexpr auto stdtuple(const char propName[], const size_type i) {
      return stdtuple_impl(_tagOffsets[propIndex(propName)], i, std::make_index_sequence<d>{});
    }

    constexpr channel_counter_type numChannels() const noexcept {
      return this->node().child(wrapv<0>{}).channel_count();
    }

    channel_counter_type N{0};
    size_type _vectorSize{0};
    const SmallString *_tagNames{nullptr};
    const channel_counter_type *_tagOffsets{nullptr};
    const channel_counter_type *_tagSizes{nullptr};
  };

  template <execspace_e ExecSpace, typename T, auto Length, typename IndexT, typename ChnT>
  constexpr decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                                 TileVector<T, Length, IndexT, ChnT> &vec) {
    return TileVectorView<ExecSpace, TileVector<T, Length, IndexT, ChnT>>{tagNames, vec};
  }
  template <execspace_e ExecSpace, typename T, auto Length, typename IndexT, typename ChnT>
  constexpr decltype(auto) proxy(TileVector<T, Length, IndexT, ChnT> &vec) {
    return TileVectorUnnamedView<ExecSpace, TileVector<T, Length, IndexT, ChnT>>{vec};
  }

}  // namespace zs