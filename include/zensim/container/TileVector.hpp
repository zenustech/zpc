#pragma once
#include "Vector.hpp"
#include "zensim/math/Vec.h"
#include "zensim/memory/Allocator.h"
#include "zensim/resource/Resource.h"
#include "zensim/tpls/magic_enum.hpp"
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

  template <typename T, auto Length = 8, typename Index = std::size_t, typename ChnCounter = char>
  struct TileVector : Inherit<Object, TileVector<T, Length, Index, ChnCounter>>,
                      aosoa_instance<Length, T, ChnCounter, Index>,
                      MemoryHandle {
    static_assert(std::is_default_constructible_v<T> && std::is_trivially_copyable_v<T>,
                  "element is not default-constructible or trivially-copyable!");
    using base_t = aosoa_instance<Length, T, ChnCounter, Index>;
    using value_type = remove_cvref_t<T>;
    using pointer = value_type *;
    using const_pointer = const pointer;
    using reference = value_type &;
    using const_reference = const value_type &;
    using channel_counter_type = ChnCounter;
    using size_type = Index;
    using difference_type = std::make_signed_t<size_type>;
    using iterator_category = std::random_access_iterator_tag;  // std::contiguous_iterator_tag;
    static constexpr auto lane_width = Length;

    constexpr MemoryHandle &base() noexcept { return static_cast<MemoryHandle &>(*this); }
    constexpr const MemoryHandle &base() const noexcept {
      return static_cast<const MemoryHandle &>(*this);
    }
    constexpr base_t &self() noexcept { return static_cast<base_t &>(*this); }
    constexpr const base_t &self() const noexcept { return static_cast<const base_t &>(*this); }

    constexpr TileVector(memsrc_e mre = memsrc_e::host, ProcID devid = -1,
                         std::size_t alignment = std::alignment_of_v<value_type>)
        : MemoryHandle{mre, devid},
          base_t{buildInstance(mre, devid, 0, 0)},
          _tags(0),
          _size{0},
          _align{alignment} {}
    TileVector(channel_counter_type numChns, size_type count = 0, memsrc_e mre = memsrc_e::host,
               ProcID devid = -1, std::size_t alignment = std::alignment_of_v<value_type>)
        : MemoryHandle{mre, devid},
          base_t{buildInstance(mre, devid, numChns, count)},
          _tags(numChns),
          _size{count},
          _align{alignment} {}
    TileVector(const std::vector<PropertyTag> &channelTags, size_type count = 0,
               memsrc_e mre = memsrc_e::host, ProcID devid = -1,
               std::size_t alignment = std::alignment_of_v<value_type>)
        : MemoryHandle{mre, devid},
          base_t{buildInstance(mre, devid, numTotalChannels(channelTags), count)},
          _tags{channelTags},
          _size{count},
          _align{alignment} {}

    ~TileVector() {
      deinitPropertyTags();
      if (head()) self().dealloc();
    }
    void deinitPropertyTags() {
      if (_tagNames) {
        auto memorySource = get_resource_manager().source(this->memspace());
        if (this->memspace() == memsrc_e::um)
          memorySource = memorySource.advisor("PREFERRED_LOCATION", this->devid());
        memorySource.deallocate(_tagNames);
        memorySource.deallocate(_tagSizes);
        memorySource.deallocate(_tagOffsets);
        _tagNames = nullptr;
        _tagSizes = nullptr;
        _tagOffsets = nullptr;
      }
    }
    void initPropertyTags(const channel_counter_type N) {
      deinitPropertyTags();
      auto memorySource = get_resource_manager().source(this->memspace());
      if (this->memspace() == memsrc_e::um)
        memorySource = memorySource.advisor("PREFERRED_LOCATION", this->devid());
      _tagNames = (SmallString *)memorySource.allocate(sizeof(SmallString) * N);
      _tagSizes = (channel_counter_type *)memorySource.allocate(sizeof(channel_counter_type) * N);
      _tagOffsets = (channel_counter_type *)memorySource.allocate(sizeof(channel_counter_type) * N);
    }

    auto numTotalChannels(const std::vector<PropertyTag> &tags) {
      tuple_element_t<1, PropertyTag> cnt = 0;
      for (std::size_t i = 0; i < tags.size(); ++i) cnt += get<1>(tags[i]);
      return cnt;
    }
    struct iterator : IteratorInterface<iterator> {
      constexpr iterator(const base_t &range, size_type idx, channel_counter_type chn = 0)
          : _range{range}, _idx{idx}, _chn{chn} {}

      constexpr reference dereference() { return _range(_idx / lane_width)(_idx % lane_width); }
      constexpr bool equal_to(iterator it) const noexcept { return it._idx == _idx; }
      constexpr void advance(difference_type offset) noexcept { _idx += offset; }
      constexpr difference_type distance_to(iterator it) const noexcept { return it._idx - _idx; }

    protected:
      base_t _range{};
      size_type _idx{0};
      channel_counter_type _chn{};
    };
    struct const_iterator : IteratorInterface<const_iterator> {
      constexpr const_iterator(const base_t &range, size_type idx, channel_counter_type chn = 0)
          : _range{range}, _idx{idx}, _chn{chn} {}

      constexpr const_reference dereference() {
        return _range(_idx / lane_width)(_idx % lane_width);
      }
      constexpr bool equal_to(const_iterator it) const noexcept { return it._idx == _idx; }
      constexpr void advance(difference_type offset) noexcept { _idx += offset; }
      constexpr difference_type distance_to(const_iterator it) const noexcept {
        return it._idx - _idx;
      }

    protected:
      base_t _range{};
      size_type _idx{0};
      channel_counter_type _chn{};
    };

    constexpr auto begin() noexcept { return make_iterator<iterator>(self(), 0); }
    constexpr auto end() noexcept { return make_iterator<iterator>(self(), size()); }
    constexpr auto begin() const noexcept { return make_iterator<const_iterator>(self(), 0); }
    constexpr auto end() const noexcept { return make_iterator<const_iterator>(self(), size()); }

    /// capacity
    constexpr size_type size() const noexcept { return _size; }
    constexpr size_type capacity() const noexcept { return self().node().extent(); }
    constexpr bool empty() noexcept { return size() == 0; }
    constexpr pointer head() const noexcept { return reinterpret_cast<pointer>(self().address()); }

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
    explicit TileVector(const TileVector &o)
        : MemoryHandle{o.base()}, _tags{o._tags}, _size{o.size()}, _align{o._align} {
      base_t tmp{buildInstance(o.memspace(), o.devid(), o.numChannels(), o.capacity())};
      if (o.size())
        copy({base(), (void *)tmp.address()}, {o.base(), o.data()},
             ds::snode_size(o.self().template node<0>()));
      self() = tmp;
    }
    TileVector &operator=(const TileVector &o) {
      if (this == &o) return *this;
      TileVector tmp{o};
      swap(tmp);
      return *this;
    }
    TileVector clone(const MemoryHandle &mh) const {
      TileVector ret{_tags, capacity(), mh.memspace(), mh.devid(), _align};
      copy({mh, (void *)ret.data()}, {base(), (void *)data()},
           ds::snode_size(self().template node<0>()));
      return ret;
    }
    /// assignment or destruction after std::move
    /// https://www.youtube.com/watch?v=ZG59Bqo7qX4
    /// explicit noexcept
    /// leave the source object in a valid (default constructed) state
    explicit TileVector(TileVector &&o) noexcept {
      const TileVector defaultVector{};
      base() = std::exchange(o.base(), defaultVector.base());
      self() = std::exchange(o.self(), defaultVector.self());
      _tags = std::exchange(o._tags, defaultVector._tags);
      _size = std::exchange(o._size, defaultVector.size());
      _align = std::exchange(o._align, defaultVector._align);
    }
    /// make move-assignment safe for self-assignment
    TileVector &operator=(TileVector &&o) noexcept {
      if (this == &o) return *this;
      TileVector tmp{std::move(o)};
      swap(tmp);
      return *this;
    }
    void swap(TileVector &o) noexcept {
      base().swap(o.base());
      std::swap(self(), o.self());
      _tags.swap(o._tags);
      std::swap(_size, o._size);
      std::swap(_align, o._align);
    }

    // constexpr operator base_t &() noexcept { return self(); }
    // constexpr operator base_t() const noexcept { return self(); }
    // void relocate(memsrc_e mre, ProcID devid) {}
    void clear() { resize(0); }
    void resize(size_type newSize) {
      const auto oldSize = size();
      if (newSize < oldSize) {
        static_assert(std::is_trivially_destructible_v<T>, "not trivially destructible");
        _size = newSize;
        return;
      }
      if (newSize > oldSize) {
        const auto oldCapacity = capacity();
        if (newSize > oldCapacity) {
          auto &rm = get_resource_manager().get();
          base_t tmp{
              buildInstance(memspace(), devid(), numChannels(), geometric_size_growth(newSize))};
          if (size())
            copy({base(), (void *)tmp.address()}, {base(), data()},
                 ds::snode_size(self().template node<0>()));
          if (oldCapacity > 0) rm.deallocate((void *)head());
          self() = tmp;
          _size = newSize;
          return;
        }
      }
    }

    constexpr channel_counter_type numChannels() const noexcept {
      return self().node().child(wrapv<0>{}).channel_count();
    }
    constexpr channel_counter_type numProperties() const noexcept { return _tags.size(); }
    constexpr auto propertyOffsets() const {
      std::vector<channel_counter_type> offsets(_tags.size());
      for (Vector<PropertyTag>::size_type i = 0; i < _tags.size(); ++i)
        offsets[i] = (i ? offsets[i - 1] + get<1>(_tags[i - 1]) : 0);
      return offsets;
    }
    void preparePropertyNames(const std::vector<SmallString> &propNames) {
      const auto N = propNames.size();
      Vector<SmallString> hostPropNames{N, memsrc_e::host, -1};
      Vector<channel_counter_type> hostPropSizes{N, memsrc_e::host, -1};
      Vector<channel_counter_type> hostPropOffsets{N, memsrc_e::host, -1};

      for (std::size_t i = 0; i < propNames.size(); ++i) hostPropNames[i] = propNames[i];
      const auto offsets = propertyOffsets();
      for (std::size_t dst = 0; dst < N; ++dst) {
        Vector<PropertyTag>::size_type i = 0;
        for (; i < _tags.size(); ++i) {
          if (zs::get<0>(_tags[i]) == propNames[dst]) {
            hostPropSizes[dst] = zs::get<1>(_tags[i]);
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
#if 0
      for (int i = 0; i < N; ++i)
        fmt::print("chk {}, {}, {}\n", hostPropNames[i], (int)hostPropOffsets[i],
                   (int)hostPropSizes[i]);
#endif
      copy({base(), _tagNames}, {{memsrc_e::host, -1}, (void *)hostPropNames.data()},
           sizeof(SmallString) * N);
      copy({base(), _tagSizes}, {{memsrc_e::host, -1}, (void *)hostPropSizes.data()},
           sizeof(channel_counter_type) * N);
      copy({base(), _tagOffsets}, {{memsrc_e::host, -1}, (void *)hostPropOffsets.data()},
           sizeof(channel_counter_type) * N);
    }
    constexpr const_pointer data() const noexcept { return (pointer)head(); }
    constexpr pointer data() noexcept { return (pointer)head(); }
    constexpr reference front() noexcept { return (*this)(0); }
    constexpr const_reference front() const noexcept { (*this)(0); }
    constexpr reference back() noexcept { return (*this)(size() - 1); }
    constexpr const_reference back() const noexcept { (*this)(size() - 1); }

    constexpr const SmallString *tagNameHandle() const noexcept { return _tagNames; }
    constexpr const channel_counter_type *tagSizeHandle() const noexcept { return _tagSizes; }
    constexpr const channel_counter_type *tagOffsetHandle() const noexcept { return _tagOffsets; }
    constexpr SmallString *tagNameHandle() noexcept { return _tagNames; }
    constexpr channel_counter_type *tagSizeHandle() noexcept { return _tagSizes; }
    constexpr channel_counter_type *tagOffsetHandle() noexcept { return _tagOffsets; }

  protected:
    constexpr auto buildInstance(memsrc_e mre, ProcID devid, channel_counter_type numChns,
                                 size_type capacity) {
      using namespace ds;
      tile_snode<lane_width, wrapt<T>, channel_counter_type> tilenode{
          ds::decorations<ds::soa>{}, static_domain<lane_width>{}, zs::make_tuple(wrapt<T>{}),
          zs::make_tuple(numChns)};
      uniform_domain<0, size_type, 1, index_seq<0>> dom{wrapv<0>{}, capacity / lane_width + 1};
      aosoa_snode<lane_width, wrapt<T>, channel_counter_type, size_type> node{
          ds::static_decorator{}, dom, zs::make_tuple(tilenode), vseq_t<1>{}};
      auto inst = instance{wrapv<dense>{}, zs::make_tuple(node)};

      if (capacity) {
        auto memorySource = get_resource_manager().source(mre);
        if (mre == memsrc_e::um) memorySource = memorySource.advisor("PREFERRED_LOCATION", devid);
        /// additional parameters should match allocator_type
        inst.alloc(memorySource);
      }
      return inst;
    }
    constexpr std::size_t geometric_size_growth(std::size_t newSize) noexcept {
      size_type geometricSize = capacity();
      geometricSize = geometricSize + geometricSize / 2;
      if (newSize > geometricSize) return newSize;
      return geometricSize;
    }
    constexpr GeneralAllocator getCurrentAllocator() {
      auto memorySource = get_resource_manager().source(this->memspace());
      if (this->memspace() == memsrc_e::um)
        memorySource = memorySource.advisor("PREFERRED_LOCATION", this->devid());
      return memorySource;
    }

    std::vector<PropertyTag> _tags;  // on host
    /// for proxy use
    SmallString *_tagNames{nullptr};
    channel_counter_type *_tagSizes{nullptr};
    channel_counter_type *_tagOffsets{nullptr};
    size_type _size{0};  // size
    size_type _align{0};
  };

  template <execspace_e, typename TileVectorT, typename = void> struct TileVectorProxy;
  template <execspace_e, typename TileVectorT, typename = void> struct TileVectorUnnamedProxy;

  template <execspace_e Space, typename TileVectorT> struct TileVectorProxy<Space, TileVectorT>
      : TileVectorT::base_t {
    using value_type = typename TileVectorT::value_type;
    using tile_vector_t = typename TileVectorT::base_t;
    using size_type = typename TileVectorT::size_type;
    using channel_counter_type = typename TileVectorT::channel_counter_type;
    static constexpr auto lane_width = TileVectorT::lane_width;

    constexpr TileVectorProxy() = default;
    ~TileVectorProxy() = default;
    explicit TileVectorProxy(const std::vector<SmallString> &tagNames, TileVectorT &tilevector)
        : tile_vector_t{tilevector.self()}, _vectorSize{tilevector.size()}, N{tagNames.size()} {
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
      for (channel_counter_type i = 0; i < RetT::extent; ++i)
        ret.data()[i] = static_cast<tile_vector_t &>(*this)(a)(chnOffset + i, b);
      return ret;
    }
    template <std::size_t... Is> constexpr auto tuple_impl(const channel_counter_type chnOffset,
                                                           const size_type i, index_seq<Is...>) {
      const auto a = i / lane_width, b = i % lane_width;
#if 0
      using Tuple = typename gen_seq<N>::template uniform_types_t<std::tuple, value_type &>;
      return Tuple{static_cast<tile_vector_t &>(*this)(a)(chnOffset + Is, b)...};
#else
      return zs::forward_as_tuple(static_cast<tile_vector_t &>(*this)(a)(chnOffset + Is, b)...);
#endif
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
  decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                       TileVector<T, Length, IndexT, ChnT> &vec) {
    return TileVectorProxy<ExecSpace, TileVector<T, Length, IndexT, ChnT>>{tagNames, vec};
  }

}  // namespace zs