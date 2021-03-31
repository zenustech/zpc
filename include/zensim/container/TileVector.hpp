#pragma once
#include <zensim/resource/Resource.h>
#include <zensim/math/Vec.h>
#include <zensim/memory/Allocator.h>
#include <zensim/types/Iterator.h>
#include <zensim/types/Polymorphism.h>
#include <zensim/types/Value.h>

#include <zensim/types/RuntimeStructurals.hpp>

#include <zensim/tpls/magic_enum.hpp>

namespace zs {

  /// channel name enum class
  enum class chn_e {
    chI = 0,
    chF,
    chL,
    chD,
    chVec3i,
    chVec3f,
    chVec3l,
    chVec3d,
    chMat3f,
    chMat3d,
    chMat4f,
    chMat4d
  };

  /// reserved for future variant-channel snode (not yet impl_ed)
  using named_channel
      = variant<wrapt<i32>, wrapt<f32>, wrapt<i64>, wrapt<f64>, wrapt<vec3i>, wrapt<vec3f>,
                wrapt<vec3l>, wrapt<vec3d>, wrapt<mat3f>, wrapt<mat3d>, wrapt<mat4f>, wrapt<mat4d>>;

  template <auto Length, typename... Chns>  ///< length should be power of 2
  using tile_snode
      = ds::snode_t<ds::decorations<ds::soa>, ds::static_domain<Length>, tuple<wrapt<Chns>...>,
                    typename gen_seq<sizeof...(Chns)>::template uniform_vseq<1>>;

  template <auto Length, typename Index, typename... Chns> using aosoa_snode
      = ds::snode_t<ds::static_decorator<>, ds::uniform_domain<0, Index, 1, index_seq<0>>,
                    tuple<tile_snode<Length, Chns...>>, vseq_t<1>>;

  template <auto Length, typename Index, typename... Attrs> using aosoa_instance
      = ds::instance_t<ds::dense, aosoa_snode<Length, Index, Attrs...>>;

  template <int Length, typename... Attrs> struct TileVector
      : Inherit<Object, TileVector<Length, Attrs...>>,
        aosoa_instance<Length, int, Attrs...>,
        MemoryHandle {
    /// using int here because there generally won't be more than 2000000000 tiles
    static constexpr int tile_size = Length;
    using channels_t = tuple<wrapt<Attrs>...>;
    using tile_snode_type = tile_snode<tile_size, Attrs...>;
    using size_type = int;
    using snode_type = aosoa_snode<tile_size, size_type, Attrs...>;
    using base_t = aosoa_instance<tile_size, size_type, Attrs...>;
    using allocator_type = typename umpire::strategy::MixedPool;

    constexpr MemoryHandle &base() noexcept { return static_cast<MemoryHandle &>(*this); }
    constexpr const MemoryHandle &base() const noexcept {
      return static_cast<const MemoryHandle &>(*this);
    }
    constexpr base_t &self() noexcept { return static_cast<base_t &>(*this); }
    constexpr const base_t &self() const noexcept { return static_cast<const base_t &>(*this); }

    TileVector(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : MemoryHandle{mre, devid}, base_t{buildInstance(mre, devid, 0)}, _size{0} {}
    TileVector(size_type count, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : MemoryHandle{mre, devid},
          base_t{buildInstance(mre, devid, count + count / 2)},
          _size{count} {}

    ~TileVector() {
      if (head()) self().dealloc();
    }

    void debug() const {
      fmt::print("procid: {}, memspace: {}, size: {}, capacity: {}\n", static_cast<int>(devid()),
                 static_cast<int>(memspace()), size(), capacity());
    }

    /// capacity

    constexpr size_type size() const noexcept { return _size; }
    constexpr size_type capacity() const noexcept { return node().extent(); }
    constexpr bool empty() noexcept { return size() == 0; }
    constexpr std::uintptr_t head() const noexcept { return self().address(); }
    constexpr std::uintptr_t tail() const noexcept { return head() + node().size(); }
    /// ctor, assignment operator
    explicit TileVector(const TileVector &o) : MemoryHandle{o.base()}, _size{o.size()} {
      auto &rm = get_resource_manager().self();
      base_t tmp{buildInstance(o.memspace(), o.devid(), o.capacity())};
      if (o.size()) rm.copy((void *)tmp.address(), (void *)o.head(), o.usedBytes());
      self() = tmp;
    }
    TileVector &operator=(const TileVector &o) {
      if (this == &o) return *this;
      TileVector tmp{o};
      swap(tmp);
      return *this;
    }
    explicit TileVector(TileVector &&o) noexcept {
      const TileVector defaultVector{};
      base() = std::exchange(o.base(), defaultVector.base());
      self() = std::exchange(o.self(), defaultVector.self());
      _size = std::exchange(o._size, defaultVector.size());
    }
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
    }

    // void relocate(memsrc_e mre, ProcID devid) {}
    void clear() { resize(0); }
    void resize(size_type newSize) {
      const auto oldSize = size();
      if (newSize < oldSize) {
        _size = newSize;
        return;
      }
      if (newSize > oldSize) {
        const auto oldCapacity = capacity();
        if (newSize > oldCapacity) {
          auto &rm = get_resource_manager().self();
          if (devid() != -1) {
            base_t tmp{buildInstance(memspace(), devid(), geometric_size_growth(newSize))};
            if (size()) rm.copy((void *)tmp.address(), (void *)head(), usedBytes());
            if (oldCapacity > 0) rm.deallocate((void *)head());

            self() = tmp;
          } else {
            /// expect this to throw if failed
            this->assign(rm.reallocate((void *)this->address(),
                                       tileSize() * geometric_size_growth(newSize),
                                       getCurrentAllocator()));
          }
          _size = newSize;
          return;
        }
      }
    }

  protected:
    constexpr snode_type &node() noexcept { return self().node(); }
    constexpr const snode_type &node() const noexcept { return self().node(); }
    constexpr std::size_t tileSize() const noexcept { return node().element_size(); }
    constexpr std::size_t usedBytes() const noexcept { return tileSize() * size(); }

    constexpr auto buildInstance(memsrc_e mre, ProcID devid, size_type capacity) {
      using namespace ds;
      constexpr auto dec = ds::static_decorator{};
      uniform_domain<0, size_type, 1, index_seq<0>> dom{wrapv<0>{}, capacity};
      snode_type node{dec, dom, zs::make_tuple(tile_snode_type{}), vseq_t<1>{}};
      auto inst = instance{wrapv<dense>{}, zs::make_tuple(node)};

      if (capacity) {
        auto memorySource = get_resource_manager().source(mre);
        if (mre == memsrc_e::um) memorySource = memorySource.advisor("PREFERRED_LOCATION", devid);
        /// additional parameters should match allocator_type
        inst.template alloc<allocator_type>(memorySource, 1 << 8, 1 << 17, 2ull << 20, 16,
                                            512ull << 20, 1 << 10, inst.maxAlignment());
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
      return memorySource.template allocator<allocator_type>(
          1 << 8, 1 << 17, 2ull << 20, 16, 512ull << 20, 1 << 10, this->maxAlignment());
    }

    size_type _size{0};  // size
  };

#if 0
template <auto Length, typename... Attrs, typename... Args>
TileVector(wrapv<Length>, const tuple<wrapt<Attrs>...> &chns, Args &&...args)
    -> TileVector<Length, Attrs...>;
#endif

}  // namespace zs