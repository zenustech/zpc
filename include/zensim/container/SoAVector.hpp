#pragma once

#include "zensim/memory/Allocator.h"
#include "zensim/resource/Resource.h"
#include "zensim/tpls/magic_enum.hpp"
#include "zensim/types/Iterator.h"
#include "zensim/types/RuntimeStructurals.hpp"

namespace zs {

  template <typename Snode, typename Chn = int, typename Index = std::size_t> using soa_vector_snode
      = ds::snode_t<ds::decorations<ds::soa>, ds::uniform_domain<0, Index, 1, index_seq<0>>,
                    tuple<Snode>, tuple<Chn>>;
  template <typename T, typename Chn = int, typename Index = std::size_t> using soa_vector_instance
      = ds::instance_t<ds::dense, soa_vector_snode<wrapt<T>, Chn, Index>>;

  template <typename T, typename ChnCounter, typename Index = std::size_t> struct SoAVector
      : Inherit<Object, SoAVector<T, ChnCounter, Index>>,
        soa_vector_instance<T, ChnCounter, Index>,
        MemoryHandle {
    static_assert(std::is_default_constructible_v<T> && std::is_trivially_copyable_v<T>,
                  "element is not default-constructible or trivially-copyable!");
    using base_t = soa_vector_instance<T, ChnCounter, Index>;
    using value_type = remove_cvref_t<T>;
    using pointer = value_type *;
    using const_pointer = const pointer;
    using reference = value_type &;
    using const_reference = const value_type &;
    using channel_counter_type = ChnCounter;
    using size_type = Index;
    using difference_type = std::make_signed_t<size_type>;
    using iterator_category = std::random_access_iterator_tag;  // std::contiguous_iterator_tag;

    constexpr MemoryHandle &base() noexcept { return static_cast<MemoryHandle &>(*this); }
    constexpr const MemoryHandle &base() const noexcept {
      return static_cast<const MemoryHandle &>(*this);
    }
    constexpr base_t &self() noexcept { return static_cast<base_t &>(*this); }
    constexpr const base_t &self() const noexcept { return static_cast<const base_t &>(*this); }

    constexpr SoAVector(memsrc_e mre = memsrc_e::host, ProcID devid = -1,
                        std::size_t alignment = std::alignment_of_v<value_type>)
        : MemoryHandle{mre, devid},
          base_t{buildInstance(mre, devid, 0, 0)},
          _size{0},
          _align{alignment} {}
    SoAVector(channel_counter_type numChns = 1, size_type count = 0, memsrc_e mre = memsrc_e::host,
              ProcID devid = -1, std::size_t alignment = std::alignment_of_v<value_type>)
        : MemoryHandle{mre, devid},
          base_t{buildInstance(mre, devid, numChns, count)},
          _size{count},
          _align{alignment} {}

    ~SoAVector() {
      if (head()) self().dealloc();
    }

    struct iterator : IteratorInterface<iterator> {
      constexpr iterator(const base_t &range, size_type idx, channel_counter_type chn = 0)
          : _range{range}, _idx{idx}, _chn{chn} {}

      constexpr reference dereference() { return _range(_chn, _idx); }
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

      constexpr const_reference dereference() { return _range(_chn, _idx); }
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
      return self()(chn, idx);
    }
    constexpr conditional_t<std::is_fundamental_v<value_type>, value_type, const_reference>
    operator[](const std::tuple<channel_counter_type, size_type> index) const noexcept {
      const auto [chn, idx] = index;
      return self()(chn, idx);
    }
    /// ctor, assignment operator
    explicit SoAVector(const SoAVector &o)
        : MemoryHandle{o.memoryHandle()}, _size{o.size()}, _align{o._align} {
      auto &rm = get_resource_manager().get();
      base_t tmp{buildInstance(o.memspace(), o.devid(), numChannels(), o.capacity())};
      if (o.size()) rm.copy((void *)tmp.address(), o.head());
      self() = tmp;
      base() = o.base();
    }
    SoAVector &operator=(const SoAVector &o) {
      if (this == &o) return *this;
      SoAVector tmp{o};
      swap(tmp);
      return *this;
    }
    SoAVector clone(const MemoryHandle &mh) {
      SoAVector ret{numChannels(), capacity(), mh.memspace(), mh.devid(), _align};
      get_resource_manager().get().copy((void *)ret.head(), (void *)head());
      return ret;
    }
    /// assignment or destruction after std::move
    /// https://www.youtube.com/watch?v=ZG59Bqo7qX4
    /// explicit noexcept
    /// leave the source object in a valid (default constructed) state
    explicit SoAVector(SoAVector &&o) noexcept {
      const SoAVector defaultVector{};
      base() = std::exchange(o.base(), defaultVector.memoryHandle());
      self() = std::exchange(o.self(), defaultVector.self());
      _size = std::exchange(o._size, defaultVector.size());
      _align = std::exchange(o._align, std::alignment_of_v<T>);
    }
    /// make move-assignment safe for self-assignment
    SoAVector &operator=(SoAVector &&o) noexcept {
      if (this == &o) return *this;
      SoAVector tmp{std::move(o)};
      swap(tmp);
      return *this;
    }
    void swap(SoAVector &o) noexcept {
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
        if constexpr (!std::is_trivially_destructible_v<T>) {
          static_assert(!std::is_trivial_v<T>, "should not activate this scope");
          pointer ed = tail();
          for (pointer e = head() + newSize; e < ed; ++e) e->~T();
        }
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
      return self().node().channel_count();
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
      constexpr auto dec = ds::static_decorator{};
      uniform_domain<0, size_type, 1, index_seq<0>> dom{wrapv<0>{}, capacity};
      soa_vector_snode<wrapt<T>, channel_counter_type, size_type> node{
          dec, dom, zs::make_tuple(wrapt<T>{}), zs::make_tuple(numChns)};
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

    size_type _size{0};  // size
    size_type _align{0};
  };

  template <execspace_e, typename SoAVectorT, typename = void> struct SoAVectorProxy;

  template <execspace_e Space, typename SoAVectorT> struct SoAVectorProxy<Space, SoAVectorT>
      : SoAVectorT::base_t {
    using soa_vector_t = typename SoAVectorT::base_t;
    using size_type = typename SoAVectorT::size_type;
    using channel_counter_type = typename SoAVectorT::channel_counter_type;

    constexpr SoAVectorProxy() = default;
    ~SoAVectorProxy() = default;
    explicit SoAVectorProxy(SoAVectorT &soavector)
        : soa_vector_t{soavector.self()}, _vectorSize{soavector.size()} {}

    constexpr channel_counter_type numChannels() const noexcept {
      return this->node().child(wrapv<0>{}).channel_count();
    }

    size_type _vectorSize;
  };

  template <execspace_e ExecSpace, typename T, typename ChnT, typename IndexT>
  decltype(auto) proxy(SoAVector<T, ChnT, IndexT> &vec) {
    return SoAVectorProxy<ExecSpace, SoAVector<T, ChnT, IndexT>>{vec};
  }

}  // namespace zs