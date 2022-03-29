#pragma once
#include "zensim/container/HashTable.hpp"
#include "zensim/container/TileVector.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/memory/Allocator.h"

namespace zs {

  struct Vec3iTable {
    static constexpr int dim = 3;
    using index_t = i32;
    using coord_t = vec<index_t, dim>;
    using key_t = u64;
    using value_t = int;  // number of (2^12) chunks, int is sufficient

    using allocator_type = ZSPmrAllocator<>;
    using size_type = std::make_unsigned_t<value_t>;
    using difference_type = std::make_signed_t<size_type>;

    struct Table {
      Table() = default;
      Table(const Table &) = default;
      Table(Table &&) noexcept = default;
      Table &operator=(const Table &) = default;
      Table &operator=(Table &&) noexcept = default;
      Table(const allocator_type &allocator, std::size_t numEntries)
          : keys{allocator, numEntries}, indices{allocator, numEntries} {}

      Vector<key_t> keys;
      Vector<value_t> indices;
    };

    static_assert(
        std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_swap::value,
        "allocator should propagate on copy, move and swap!");

    static constexpr index_t key_scalar_sentinel_v = ~(index_t)0;
    static constexpr value_t sentinel_v{(value_t)-1};  // this requires value_t to be signed type
    static constexpr std::size_t reserve_ratio_v = 16;

    constexpr decltype(auto) memoryLocation() const noexcept { return _allocator.location; }
    constexpr ProcID devid() const noexcept { return memoryLocation().devid(); }
    constexpr memsrc_e memspace() const noexcept { return memoryLocation().memspace(); }
    decltype(auto) get_allocator() const noexcept { return _allocator; }

    constexpr auto &self() noexcept { return _table; }
    constexpr const auto &self() const noexcept { return _table; }

    constexpr std::size_t evaluateTableSize(std::size_t entryCnt) const {
      if (entryCnt == 0) return (std::size_t)0;
      return next_2pow(entryCnt) * reserve_ratio_v;
    }
    Vec3iTable(const allocator_type &allocator, std::size_t tableSize)
        : _table{allocator, evaluateTableSize(tableSize)},
          _allocator{allocator},
          _tableSize{static_cast<value_t>(evaluateTableSize(tableSize))},
          _cnt{allocator, 1},
          _activeKeys{allocator, evaluateTableSize(tableSize)} {
      value_t res[1];
      res[0] = (value_t)0;
      Resource::copy(MemoryEntity{_cnt.memoryLocation(), (void *)_cnt.data()},
           MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)res}, sizeof(value_t));
    }
    Vec3iTable(std::size_t tableSize, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : Vec3iTable{get_memory_source(mre, devid), tableSize} {}
    Vec3iTable(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : Vec3iTable{get_memory_source(mre, devid), (std::size_t)0} {}

    ~Vec3iTable() = default;

    Vec3iTable(const Vec3iTable &o)
        : _table{o._table},
          _allocator{o._allocator},
          _tableSize{o._tableSize},
          _cnt{o._cnt},
          _activeKeys{o._activeKeys} {}
    Vec3iTable &operator=(const Vec3iTable &o) {
      if (this == &o) return *this;
      Vec3iTable tmp(o);
      swap(tmp);
      return *this;
    }
    Vec3iTable clone(const allocator_type &allocator) const {
      Vec3iTable ret{allocator, _tableSize / reserve_ratio_v};
      if (_cnt.size() > 0)
        Resource::copy(MemoryEntity{ret._cnt.memoryLocation(), (void *)ret._cnt.data()},
             MemoryEntity{_cnt.memoryLocation(), (void *)_cnt.data()}, sizeof(value_t));
      if (_activeKeys.size() > 0)
        Resource::copy(MemoryEntity{ret._activeKeys.memoryLocation(), (void *)ret._activeKeys.data()},
             MemoryEntity{_activeKeys.memoryLocation(), (void *)_activeKeys.data()},
             sizeof(key_t) * _activeKeys.size());
      if (_tableSize > 0) {
        ret.self().keys = self().keys.clone(allocator);
        ret.self().indices = self().indices.clone(allocator);
      }
      return ret;
    }
    Vec3iTable clone(const MemoryLocation &mloc) const {
      return clone(get_memory_source(mloc.memspace(), mloc.devid()));
    }

    Vec3iTable(Vec3iTable &&o) noexcept {
      const Vec3iTable defaultTable{};
      self() = std::exchange(o.self(), defaultTable.self());
      _allocator = std::exchange(o._allocator, defaultTable._allocator);
      _tableSize = std::exchange(o._tableSize, defaultTable._tableSize);
      _cnt = std::exchange(o._cnt, defaultTable._cnt);
      _activeKeys = std::exchange(o._activeKeys, defaultTable._activeKeys);
    }
    Vec3iTable &operator=(Vec3iTable &&o) noexcept {
      if (this == &o) return *this;
      Vec3iTable tmp(std::move(o));
      swap(tmp);
      return *this;
    }
    void swap(Vec3iTable &o) noexcept {
      std::swap(self(), o.self());
      std::swap(_allocator, o._allocator);
      std::swap(_tableSize, o._tableSize);
      std::swap(_cnt, o._cnt);
      std::swap(_activeKeys, o._activeKeys);
    }

    inline value_t size() const {
      value_t res[1];
      Resource::copy(MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)res},
           MemoryEntity{_cnt.memoryLocation(), (void *)_cnt.data()}, sizeof(value_t));
      return res[0];
    }

    template <typename Policy> void resize(Policy &&, std::size_t tableSize);

    Table _table;
    allocator_type _allocator;
    value_t _tableSize;
    Vector<value_t> _cnt;
    Vector<key_t> _activeKeys;
  };
#if 0
  template <typename Vec3iTableView> struct Vec3iTableCopy {
    using size_type = typename Vec3iTableView::size_type;
    Vec3iTableCopy(Vec3iTableView src, Vec3iTableView dst) : src{src}, dst{dst} {}
    constexpr void operator()(size_type i) {
    }
    Vec3iTableView src, dst;
  };
  template <typename Policy> void Vec3iTable::resize(Policy &&policy, std::size_t tableSize) {
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    const auto s = size();
    Vec3iTable tmp{get_allocator(), tableSize};
    policy(range(s), Vec3iTableReinsert{proxy<space>(*this), proxy<space>(tmp)});
    *this = std::move(tmp);
  }
#endif
  template <execspace_e space, typename Vec3iTableT> struct Vec3iTableView {
    static constexpr bool is_const_structure = std::is_const_v<Vec3iTableT>;
    using vec3i_table_t = std::remove_const_t<Vec3iTableT>;
    static constexpr int dim = vec3i_table_t::dim;
    static constexpr auto exectag = wrapv<space>{};
    using index_t = typename vec3i_table_t::index_t;
    using coord_t = typename vec3i_table_t::coord_t;
    using key_t = typename vec3i_table_t::key_t;
    using value_t = typename vec3i_table_t::value_t;
    using unsigned_value_t = std::make_unsigned_t<value_t>;

    using size_type = typename vec3i_table_t::size_type;
    using difference_type = typename vec3i_table_t::difference_type;

    static constexpr index_t key_sentinel_v{vec3i_table_t::key_scalar_sentinel_v};
    static constexpr value_t sentinel_v{vec3i_table_t::sentinel_v};

    struct table_t {
      key_t *keys;
      value_t *indices;
    };

    constexpr Vec3iTableView() = default;
    ~Vec3iTableView() = default;

    explicit constexpr Vec3iTableView(Vec3iTableT &table)
        : _table{table.self().keys.data(), table.self().indices.data()},
          _tableSize{table._tableSize},
          _cnt{table._cnt.data()},
          _activeKeys{table._activeKeys.data()} {}

    constexpr u64 coord_to_key(const coord_t &coord) const noexcept {
      return ((u64)(coord[0] & ~(index_t)((1 << 12) - 1)) << (u64)28)
             | ((u64)(coord[1] & ~(index_t)((1 << 12) - 1)) << (u64)8)
             | (((u64)(coord[2] & ~(index_t)((1 << 12) - 1))) >> (u64)12);
    }
    constexpr u32 coord_hash(const coord_t &coord) const noexcept {
      return ((1 << 20) - 1) & (coord[0] * 73856093 ^ coord[1] * 19349663 ^ coord[2] * 83492791);
    }
    template <bool Pred = !is_const_structure>
    constexpr std::enable_if_t<Pred, value_t> insert(const coord_t &coord) {
      using namespace placeholders;
      key_t key = coord_to_key(coord);
      key_t hashedentry = key % _tableSize;
      key_t storedKey = atomic_cas(exectag, &_table.keys[hashedentry], key_sentinel_v, key);
      for (; !(storedKey == key_sentinel_v || storedKey == key);) {
        hashedentry = (hashedentry + 127) % _tableSize;
        storedKey = atomic_cas(exectag, &_table.keys[hashedentry], key_sentinel_v, key);
      }
      if (storedKey == key_sentinel_v) {
        auto localno = (value_t)atomic_add(exectag, (unsigned_value_t *)_cnt, (unsigned_value_t)1);
        _table.indices[hashedentry] = localno;
        _activeKeys[localno] = key;
        if (localno >= _tableSize - 20)
          printf("proximity!!! %d -> %d\n", (int)localno, (int)_tableSize);
        return localno;  ///< only the one that inserts returns the actual index
      }
      return sentinel_v;
    }
    /// make sure no one else is inserting in the same time!
    constexpr value_t query(const coord_t &coord) const {
      using namespace placeholders;
      key_t key = coord_to_key(coord);
      key_t hashedentry = key % _tableSize;
      while (true) {
        if (key == (key_t)_table.keys[hashedentry]) return _table.indices[hashedentry];
        if (_table.indices[hashedentry] == sentinel_v) return sentinel_v;
        hashedentry += 127;  ///< search next entry
        if (hashedentry > _tableSize) hashedentry = hashedentry % _tableSize;
      }
    }
    template <execspace_e S = space, enable_if_t<S == execspace_e::host> = 0> void clear() {
      using namespace placeholders;
      // reset counter
      *_cnt = 0;
      // reset table
      for (key_t entry = 0; entry < _tableSize; ++entry) {
        _table.keys[entry] = key_sentinel_v;
        _table.indices[entry] = sentinel_v;
      }
    }

    table_t _table;
    const value_t _tableSize;
    value_t *_cnt;
    key_t *_activeKeys;
  };

  template <typename T> struct SparseGrid {
    using value_type = T;
    using size_type = std::size_t;
    using mr_allocator_type = ZSPmrAllocator<>;
    using vmr_allocator_type = ZSPmrAllocator<true>;
    using index_type = i32;

    static constexpr int dim = 3;
    using coord_t = vec<index_type, dim>;

    SparseGrid(size_type numChunks = 512, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : _rootDicts{numChunks, mre, devid},
          _numChannels{1},
          _sparseBlocks{mre, devid},
          _forests{} {}

    struct RootNode {
      // log2 of voxel count in a chunk in one dimension
      static constexpr std::size_t log2dim = 12;  // 3 + 4 + 5
      // total voxel count in a chunk
      static constexpr size_type num_voxels = (size_type)1 << (size_type)(dim * log2dim);
      static constexpr std::size_t num_channels = 128;
      static constexpr std::size_t total_data_bytes
          = num_channels * num_voxels * sizeof(value_type);
      static constexpr std::size_t total_mask_bytes = num_voxels / sizeof(char);
      static constexpr std::size_t block_side_length = 8;
      static constexpr std::size_t block_size
          = block_side_length * block_side_length * block_side_length;

      static constexpr size_type linear_offset(const coord_t &coord) noexcept {
        return (expand_bits_64(coord[0] & ((1 << log2dim) - 1)) << 2)
               | (expand_bits_64(coord[1] & ((1 << log2dim) - 1)) << 1)
               | (expand_bits_64(coord[2] & ((1 << log2dim) - 1)));
      }

      struct DataArena {
        DataArena(memsrc_e mre = memsrc_e::host, ProcID did = -1) {
          const mem_tags tag = to_memory_source_tag(mre);
          _allocator.setOwningUpstream<arena_virtual_memory_resource>(tag, did, total_data_bytes);
        }

        vmr_allocator_type _allocator;
      };
      struct MaskArena {
        MaskArena(memsrc_e mre = memsrc_e::host, ProcID did = -1) {
          const mem_tags tag = to_memory_source_tag(mre);
          _allocator.setOwningUpstream<arena_virtual_memory_resource>(tag, did, total_mask_bytes);
        }

        vmr_allocator_type _allocator;
      };

      // make container vmr first?
      RootNode(memsrc_e mre = memsrc_e::host, ProcID did = -1)
          : _dataArena{mre, did}, _maskArena{mre, did} {}

      ~RootNode() = default;
      // disable copy for the moment, before vmr-based container impl
      RootNode(const RootNode &) = delete;
      RootNode(RootNode &&) = default;

      // Vector<value_type> _dataGrid;
      // Vector<u64> _activeMasks;  // masks for 'num_voxels' voxels
      DataArena _dataArena{};
      MaskArena _maskArena{};
    };

    Vec3iTable _rootDicts;  /// coord -> index
    size_type _numChannels;
    Vector<coord_t> _sparseBlocks;
    std::vector<RootNode> _forests;
  };
  /// 1 recompute sparsity pass 0 (compute device): [coords -> _sparseBlocks]
  /// 2 activation pass (host): [_sparseBlocks, nchns -> _rootNodes (evict + insert + resize chns)]
  ///  prepare rootnode, allocate mask&data memory in each rootnode
  /// 3 access pass (compute device): [view -> _dataArena]

#if 0
  template <typename T> struct AdaptiveGrid {
    using value_type = T;
    using size_type = std::size_t;
    using mr_allocator_type = ZSPmrAllocator<>;
    using vmr_allocator_type = ZSPmrAllocator<true>;
    using index_type = int;

    static constexpr int dim = 3;
    using coord = vec<index_type, dim>;

    struct RootNode {
      static constexpr int num_levels = 3;
      static constexpr std::size_t num_level_bits[num_levels] = {3, 4, 5};
      /// ...
    };

    Vector<RootNode> _rootNodes;
    Vec3iTable _rootDicts;  /// coord -> index
  };
#endif

}  // namespace zs