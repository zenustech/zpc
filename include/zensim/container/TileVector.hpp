#pragma once
#include "Vector.hpp"
#include "zensim/math/Vec.h"
#include "zensim/memory/Allocator.h"
#include "zensim/resource/Resource.h"
#include "zensim/tpls/magic_enum.hpp"
#include "zensim/types/Iterator.h"
#include "zensim/types/Polymorphism.h"
#include "zensim/types/RuntimeStructurals.hpp"

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
          _tags{mre, devid, alignment},
          _size{0},
          _align{alignment} {}
    TileVector(channel_counter_type numChns = 1, size_type count = 0, memsrc_e mre = memsrc_e::host,
               ProcID devid = -1, std::size_t alignment = std::alignment_of_v<value_type>)
        : MemoryHandle{mre, devid},
          base_t{buildInstance(mre, devid, numChns, count)},
          _tags{numChns, mre, devid, alignment},
          _size{count},
          _align{alignment} {}
    TileVector(Vector<PropertyTag> channelTags, size_type count = 0, memsrc_e mre = memsrc_e::host,
               ProcID devid = -1, std::size_t alignment = std::alignment_of_v<value_type>)
        : MemoryHandle{mre, devid},
          base_t{buildInstance(mre, devid, numTotalChannels(channelTags), count)},
          _tags{channelTags.clone(MemoryHandle{mre, devid})},
          _size{count},
          _align{alignment} {}

    ~TileVector() {
      if (head()) self().dealloc();
    }

    auto numTotalChannels(const Vector<PropertyTag> &tags) {
      return tags.size();
#if 0
      tuple_element_t<1, PropertyTag> cnt = 0;
      for (Vector<PropertyTag>::size_type i = 0; i < tags.size(); ++i) cnt += get<1>(tags[i]);
      return cnt;
#endif
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
        : MemoryHandle{o.memoryHandle()}, _tags{o._tags}, _size{o.size()}, _align{o._align} {
      auto &rm = get_resource_manager().get();
      base_t tmp{buildInstance(o.memspace(), o.devid(), numChannels(), o.capacity())};
      if (o.size()) rm.copy((void *)tmp.address(), o.head());
      self() = tmp;
      base() = o.base();
    }
    TileVector &operator=(const TileVector &o) {
      if (this == &o) return *this;
      TileVector tmp{o};
      swap(tmp);
      return *this;
    }
    TileVector clone(const MemoryHandle &mh) {
      TileVector ret{_tags, capacity(), mh.memspace(), mh.devid(), _align};
      get_resource_manager().get().copy((void *)ret.head(), (void *)head());
      return ret;
    }
    /// assignment or destruction after std::move
    /// https://www.youtube.com/watch?v=ZG59Bqo7qX4
    /// explicit noexcept
    /// leave the source object in a valid (default constructed) state
    explicit TileVector(TileVector &&o) noexcept {
      const TileVector defaultVector{};
      base() = std::exchange(o.base(), defaultVector.memoryHandle());
      self() = std::exchange(o.self(), defaultVector.self());
      _size = std::exchange(o._size, defaultVector.size());
      _align = std::exchange(o._align, std::alignment_of_v<T>);
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
          if (devid() != -1) {
            base_t tmp{
                buildInstance(memspace(), devid(), numChannels(), geometric_size_growth(newSize))};
            if (size()) rm.copy((void *)tmp.address(), (void *)head());
            if (oldCapacity > 0) rm.deallocate((void *)head());

            self() = tmp;
          } else {
            /// expect this to throw if failed
            this->assign(rm.reallocate((void *)this->address(),
                                       sizeof(T) * geometric_size_growth(newSize),
                                       getCurrentAllocator()));
          }
          _size = newSize;
          return;
        }
      }
    }

    constexpr channel_counter_type numChannels() const noexcept {
      return self().node().child(wrapv<0>{}).channel_count();
    }
    constexpr const_pointer data() const noexcept { return (pointer)head(); }
    constexpr pointer data() noexcept { return (pointer)head(); }
    constexpr reference front() noexcept { return (*this)(0); }
    constexpr const_reference front() const noexcept { (*this)(0); }
    constexpr reference back() noexcept { return (*this)(size() - 1); }
    constexpr const_reference back() const noexcept { (*this)(size() - 1); }

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

    Vector<PropertyTag> _tags;
    size_type _size{0};  // size
    size_type _align{0};
  };

  template <execspace_e, typename TileVectorT, typename = void> struct TileVectorProxy;

  template <execspace_e Space, typename TileVectorT> struct TileVectorProxy<Space, TileVectorT>
      : TileVectorT::base_t {
    using tile_vector_t = typename TileVectorT::base_t;
    using size_type = typename TileVectorT::size_type;
    using channel_counter_type = typename TileVectorT::channel_counter_type;

    constexpr TileVectorProxy() = default;
    ~TileVectorProxy() = default;
    explicit TileVectorProxy(TileVectorT &tilevector)
        : tile_vector_t{tilevector.self()}, _vectorSize{tilevector.size()} {}

    constexpr channel_counter_type numChannels() const noexcept {
      return this->node().child(wrapv<0>{}).channel_count();
    }

    size_type _vectorSize;
  };

  template <execspace_e ExecSpace, auto Length, typename T, typename ChnT, typename IndexT>
  decltype(auto) proxy(TileVector<T, Length, IndexT, ChnT> &vec) {
    return TileVectorProxy<ExecSpace, TileVector<T, Length, IndexT, ChnT>>{vec};
  }

}  // namespace zs