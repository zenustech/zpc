#pragma once
#include "Vector.hpp"
#include "zensim/execution/Atomics.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/execution/Intrinsics.hpp"
#include "zensim/math/Hash.hpp"
#include "zensim/math/Vec.h"
#include "zensim/math/bit/Bits.h"
#include "zensim/memory/MemoryResource.h"
#include "zensim/resource/Resource.h"
#include "zensim/types/Iterator.h"

namespace zs {

  template <typename Tn_, int dim_, typename Index> struct HashTable {
    static_assert(is_same_v<Tn_, remove_cvref_t<Tn_>>, "Key is not cvref-unqualified type!");
    static_assert(std::is_default_constructible_v<Tn_>, "Key is not default-constructible!");
    static_assert(std::is_trivially_copyable_v<Tn_>, "Key is not trivially-copyable!");

    static constexpr int dim = dim_;
    using Tn = std::make_signed_t<Tn_>;
    using key_t = vec<Tn, dim>;
    using value_t = std::make_signed_t<Index>;
    using status_t = int;

    using value_type = key_t;
    using allocator_type = ZSPmrAllocator<>;
    using size_type = std::make_unsigned_t<value_t>;
    using difference_type = std::make_signed_t<size_type>;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;

    struct Table {
      Table() = default;
      Table(const Table&) = default;
      Table(Table&&) noexcept = default;
      Table &operator=(const Table &) = default;
      Table &operator=(Table &&) noexcept = default;
      Table(const allocator_type &allocator, size_type numEntries) : 
        keys{allocator, numEntries}, 
        indices{allocator, numEntries}, 
        status{allocator, numEntries} {}

      Vector<key_t> keys;
      Vector<value_t> indices;
      Vector<status_t> status;
    };

    static_assert(
        std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_swap::value,
        "allocator should propagate on copy, move and swap (for impl simplicity)!");

    static constexpr Tn key_scalar_sentinel_v = std::numeric_limits<Tn>::max();
    static constexpr value_t sentinel_v{-1};  // this requires value_t to be signed type
    static constexpr status_t status_sentinel_v{-1};
    static constexpr std::size_t reserve_ratio_v = 16;

    constexpr decltype(auto) memoryLocation() noexcept { return _allocator.location; }
    constexpr decltype(auto) memoryLocation() const noexcept { return _allocator.location; }
    constexpr ProcID devid() const noexcept { return memoryLocation().devid(); }
    constexpr memsrc_e memspace() const noexcept { return memoryLocation().memspace(); }
    constexpr decltype(auto) allocator() const noexcept { return _allocator; }

    constexpr auto &self() noexcept { return _table; }
    constexpr const auto &self() const noexcept { return _table; }

    constexpr std::size_t evaluateTableSize(std::size_t entryCnt) const {
      if (entryCnt == 0) return (std::size_t)0;
      return next_2pow(entryCnt) * reserve_ratio_v;
    }
    HashTable(const allocator_type &allocator, std::size_t tableSize)
        : _table{allocator, evaluateTableSize(tableSize)}, 
          _allocator{allocator},
          _tableSize{static_cast<value_t>(evaluateTableSize(tableSize))},
          _cnt{allocator, 1},
          _activeKeys{allocator, evaluateTableSize(tableSize)} {
      Vector<value_t> res{1};
      res[0] = (value_t)0;
      copy(MemoryEntity{_cnt.memoryLocation(), (void *)_cnt.data()}, 
           MemoryEntity{res.memoryLocation(), (void *)res.data()}, sizeof(value_t));
    }
    HashTable(std::size_t tableSize, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : HashTable{get_memory_source(mre, devid), tableSize} {}
    HashTable(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : HashTable{get_memory_source(mre, devid), (std::size_t)0} {}

    ~HashTable() = default;

    HashTable(const HashTable &o)
        : _table{o._table}, 
          _allocator{o._allocator},
          _tableSize{o._tableSize},
          _cnt{o._cnt},
          _activeKeys{o._activeKeys} {}
    HashTable &operator=(const HashTable &o) {
      if (this == &o) return *this;
      HashTable tmp(o);
      swap(tmp);
      return *this;
    }
    HashTable clone(const allocator_type &allocator) const {
      HashTable ret{allocator, _tableSize / reserve_ratio_v};
      if (_cnt.size() > 0)
        copy(MemoryEntity{ret._cnt.memoryLocation(), (void *)ret._cnt.data()},
             MemoryEntity{_cnt.memoryLocation(), (void *)_cnt.data()}, sizeof(value_t));
      if (_activeKeys.size() > 0)
        copy(MemoryEntity{ret._activeKeys.memoryLocation(), (void *)ret._activeKeys.data()},
             MemoryEntity{_activeKeys.memoryLocation(), (void *)_activeKeys.data()},
             sizeof(key_t) * _activeKeys.size());
      if (_tableSize > 0) {
        ret.self().keys = self().keys.clone(allocator);
        ret.self().indices = self().indices.clone(allocator);
        ret.self().status = self().status.clone(allocator);
      }
      return ret;
    }
    HashTable clone(const MemoryLocation &mloc) const {
      return clone(get_memory_source(mloc.memspace(), mloc.devid()));
    }

    HashTable(HashTable &&o) noexcept {
      const HashTable defaultTable{};
      self() = std::exchange(o.self(), defaultTable.self());
      _allocator = std::exchange(o._allocator, defaultTable._allocator);
      _tableSize = std::exchange(o._tableSize, defaultTable._tableSize);
      /// critical! use user-defined move assignment constructor!
      _cnt = std::exchange(o._cnt, defaultTable._cnt);
      _activeKeys = std::exchange(o._activeKeys, defaultTable._activeKeys);
    }
    HashTable &operator=(HashTable &&o) noexcept {
      if (this == &o) return *this;
      HashTable tmp(std::move(o));
      swap(tmp);
      return *this;
    }
    void swap(HashTable &o) noexcept {
      std::swap(self(), o.self());
      std::swap(_allocator, o._allocator);
      std::swap(_tableSize, o._tableSize);
      std::swap(_cnt, o._cnt);
      std::swap(_activeKeys, o._activeKeys);
    }
    friend void swap(HashTable &a, HashTable &b) { a.swap(b); }

    inline value_t size() const {
      Vector<value_t> res{1};
      copy(MemoryEntity{res.memoryLocation(), (void *)res.data()},
           MemoryEntity{_cnt.memoryLocation(), (void *)_cnt.data()}, sizeof(value_t));
      return res[0];
    }

    struct iterator_impl : IteratorInterface<iterator_impl> {
      template <typename Ti> constexpr iterator_impl(key_t *base, Ti &&idx)
          : _base{base}, _idx{static_cast<size_type>(idx)} {}

      constexpr reference dereference() { return _base[_idx]; }
      constexpr bool equal_to(iterator_impl it) const noexcept { return it._idx == _idx; }
      constexpr void advance(difference_type offset) noexcept { _idx += offset; }
      constexpr difference_type distance_to(iterator_impl it) const noexcept {
        return it._idx - _idx;
      }

    protected:
      value_type *_base{nullptr};
      size_type _idx{0};
    };
    struct const_iterator_impl : IteratorInterface<const_iterator_impl> {
      template <typename Ti> constexpr const_iterator_impl(const key_t *base, Ti &&idx)
          : _base{base}, _idx{static_cast<size_type>(idx)} {}

      constexpr const_reference dereference() { return _base[_idx]; }
      constexpr bool equal_to(const_iterator_impl it) const noexcept { return it._idx == _idx; }
      constexpr void advance(difference_type offset) noexcept { _idx += offset; }
      constexpr difference_type distance_to(const_iterator_impl it) const noexcept {
        return it._idx - _idx;
      }

    protected:
      value_type const *_base{nullptr};
      size_type _idx{0};
    };
    using iterator = LegacyIterator<iterator_impl>;
    using const_iterator = LegacyIterator<const_iterator_impl>;

    constexpr auto begin() noexcept { return make_iterator<iterator_impl>(_activeKeys.data(), 0); }
    constexpr auto end() noexcept {
      return make_iterator<iterator_impl>(_activeKeys.data(), size());
    }
    constexpr auto begin() const noexcept {
      return make_iterator<const_iterator_impl>(_activeKeys.data(), 0);
    }
    constexpr auto end() const noexcept {
      return make_iterator<const_iterator_impl>(_activeKeys.data(), size());
    }

    Table _table;
    allocator_type _allocator;
    value_t _tableSize;
    Vector<value_t> _cnt;
    Vector<key_t> _activeKeys;
  };

#if 0
  using GeneralHashTable = variant<HashTable<i32, 2, int>, HashTable<i32, 2, long long int>,
                                   HashTable<i32, 3, int>, HashTable<i32, 3, long long int>>;
#else
  using GeneralHashTable = variant<HashTable<i32, 3, int>>;
#endif

  template <execspace_e, typename HashTableT, typename = void> struct HashTableView;

  /// proxy to work within each backends
  template <execspace_e space, typename HashTableT> struct HashTableView<space, HashTableT> {
    static constexpr int dim = HashTableT::dim;
    static constexpr auto exectag = wrapv<space>{};
    using Tn = typename HashTableT::Tn;
    using key_t = typename HashTableT::key_t;
    using value_t = typename HashTableT::value_t;
    using unsigned_value_t = std::make_unsigned_t<value_t>;
    using status_t = typename HashTableT::status_t;
    struct table_t {
      key_t* keys;
      value_t* indices;
      status_t* status;
    };

    constexpr HashTableView() = default;
    ~HashTableView() = default;

    explicit constexpr HashTableView(HashTableT &table)
        : _table{table.self().keys.data(), table.self().indices.data(), table.self().status.data()},
          _tableSize{table._tableSize},
          _cnt{table._cnt.data()},
          _activeKeys{table._activeKeys.data()} {}

#if defined(__CUDACC__)
    template <execspace_e S = space, enable_if_t<S == execspace_e::cuda> = 0> 
    __forceinline__ __device__ value_t insert(const key_t &key) {
      using namespace placeholders;
      constexpr key_t key_sentinel_v = key_t::uniform(HashTableT::key_scalar_sentinel_v);
      value_t hashedentry = (do_hash(key) % _tableSize + _tableSize) % _tableSize;
      key_t storedKey = atomicKeyCAS(&_table.status[hashedentry], &_table.keys[hashedentry], key);
      for (; !(storedKey == key_sentinel_v || storedKey == key);) {
        hashedentry = (hashedentry + 127) % _tableSize;
        storedKey = atomicKeyCAS(&_table.status[hashedentry], &_table.keys[hashedentry], key);
      }
      if (storedKey == key_sentinel_v) {
        auto localno = (value_t)atomic_add(exectag, (unsigned_value_t *)_cnt, (unsigned_value_t)1);
        _table.indices[hashedentry] = localno;
        _activeKeys[localno] = key;
        if (localno >= _tableSize - 20)
          printf("proximity!!! %d -> %d\n", (int)localno, (int)_tableSize);
        return localno;  ///< only the one that inserts returns the actual index
      }
      return HashTableT::sentinel_v;
    }
#endif
    template <execspace_e S = space, enable_if_t<S != execspace_e::cuda> = 0> 
    inline value_t insert(const key_t &key) {
      using namespace placeholders;
      constexpr key_t key_sentinel_v = key_t::uniform(HashTableT::key_scalar_sentinel_v);
      value_t hashedentry = (do_hash(key) % _tableSize + _tableSize) % _tableSize;
      key_t storedKey = atomicKeyCAS(&_table.status[hashedentry], &_table.keys[hashedentry], key);
      for (; !(storedKey == key_sentinel_v || storedKey == key);) {
        hashedentry = (hashedentry + 127) % _tableSize;
        storedKey = atomicKeyCAS(&_table.status[hashedentry], &_table.keys[hashedentry], key);
      }
      if (storedKey == key_sentinel_v) {
        auto localno = (value_t)atomic_add(exectag, (unsigned_value_t *)_cnt, (unsigned_value_t)1);
        _table.indices[hashedentry] = localno;
        _activeKeys[localno] = key;
        if (localno >= _tableSize - 20)
          printf("proximity!!! %d -> %d\n", (int)localno, (int)_tableSize);
        return localno;  ///< only the one that inserts returns the actual index
      }
      return HashTableT::sentinel_v;
    }
    /// make sure no one else is inserting in the same time!
    constexpr value_t query(const key_t &key) const {
      using namespace placeholders;
      value_t hashedentry = (do_hash(key) % _tableSize + _tableSize) % _tableSize;
      while (true) {
        if (key == (key_t)_table.keys[hashedentry]) return _table.indices[hashedentry];
        if (_table.indices[hashedentry] == HashTableT::sentinel_v) return HashTableT::sentinel_v;
        hashedentry += 127;  ///< search next entry
        if (hashedentry > _tableSize) hashedentry = hashedentry % _tableSize;
      }
    }
    template <execspace_e S = space, enable_if_t<S == execspace_e::host> = 0> void clear() {
      using namespace placeholders;
      // reset counter
      *_cnt = 0;
      // reset table
      constexpr key_t key_sentinel_v = key_t::uniform(HashTableT::key_scalar_sentinel_v);
      for (value_t entry = 0; entry < _tableSize; ++entry) {
        _table.keys[entry] = key_sentinel_v;
        _table.indices[entry] = HashTableT::sentinel_v;
        _table.status[entry] = HashTableT::status_sentinel_v;
      }
    }

    table_t _table;
    const value_t _tableSize;
    value_t *_cnt;
    key_t *_activeKeys;

  protected:
    constexpr value_t do_hash(const key_t &key) const {
      std::size_t ret = key[0];
      for (int d = 1; d < HashTableT::dim; ++d) hash_combine(ret, key[d]);
      return static_cast<value_t>(ret);
    }
#if defined(__CUDACC__)
    template <execspace_e S = space, enable_if_t<S == execspace_e::cuda> = 0>
    __forceinline__ __device__ key_t atomicKeyCAS(status_t *lock, volatile key_t *const dest, const key_t &val) {
      constexpr auto execTag = wrapv<S>{};
      using namespace placeholders;
      constexpr key_t key_sentinel_v = key_t::uniform(HashTableT::key_scalar_sentinel_v);
      key_t return_val{};
      int done = 0;
      unsigned int mask = active_mask(execTag);             // __activemask();
      unsigned int active = ballot_sync(execTag, mask, 1);  //__ballot_sync(mask, 1);
      unsigned int done_active = 0;
      while (active != done_active) {
        if (!done) {
          if (atomic_cas(execTag, lock, HashTableT::status_sentinel_v, (status_t)0)
              == HashTableT::status_sentinel_v) {
            thread_fence(execTag);  // __threadfence();
            /// <deprecating volatile - JF Bastien - CppCon2019>
            /// access non-volatile using volatile semantics
            /// use cast
            (void)(return_val = *const_cast<key_t *>(dest));
            /// https://github.com/kokkos/kokkos/commit/2fd9fb04a94ecba29a04a0894c99e1d9c16ad66a
            if (return_val == key_sentinel_v) {
              for (int d = 0; d < dim; ++d) (void)(dest->data()[d] = val[d]);
              // (void)(*dest = val);
            }
            thread_fence(execTag);  // __threadfence();
            atomic_exch(execTag, lock, HashTableT::status_sentinel_v);
            done = 1;
          }
        }
        done_active = ballot_sync(execTag, mask, done);  //__ballot_sync(mask, done);
      }
      return return_val;
    }
#endif
    template <execspace_e S = space, enable_if_t<S != execspace_e::cuda> = 0>
    inline key_t atomicKeyCAS(status_t *lock, volatile key_t *const dest, const key_t &val) {
      constexpr auto execTag = wrapv<S>{};
      using namespace placeholders;
      constexpr key_t key_sentinel_v = key_t::uniform(HashTableT::key_scalar_sentinel_v);
      key_t return_val{};
      bool done = false;
      while (!done) {
        if (atomic_cas(execTag, lock, HashTableT::status_sentinel_v, (status_t)0)
            == HashTableT::status_sentinel_v) {
          (void)(return_val = *const_cast<key_t *>(dest));
          if (return_val == key_sentinel_v)
            for (int d = 0; d < dim; ++d) (void)(dest->data()[d] = val[d]);
          atomic_exch(execTag, lock, HashTableT::status_sentinel_v);
          done = true;
        }
      }
      return return_val;
    }
  };
  template <execspace_e space, typename HashTableT> struct HashTableView<space, const HashTableT> {
    static constexpr int dim = HashTableT::dim;
    static constexpr auto exectag = wrapv<space>{};
    using Tn = typename HashTableT::Tn;
    using key_t = typename HashTableT::key_t;
    using value_t = typename HashTableT::value_t;
    using unsigned_value_t = std::make_unsigned_t<value_t>;
    using status_t = typename HashTableT::status_t;
    struct table_t {
      const key_t* keys;
      const value_t* indices;
      const status_t* status;
    };

    constexpr HashTableView() = default;
    ~HashTableView() = default;

    explicit constexpr HashTableView(const HashTableT &table)
        : _table{table.self().keys.data(), table.self().indices.data(), table.self().status.data()},
          _tableSize{table._tableSize},
          _cnt{table._cnt.data()},
          _activeKeys{table._activeKeys.data()} {}

    constexpr value_t query(const key_t &key) const {
      using namespace placeholders;
      value_t hashedentry = (do_hash(key) % _tableSize + _tableSize) % _tableSize;
      while (true) {
        if (key == (key_t)_table.keys[hashedentry]) return _table.indices[hashedentry];
        if (_table.indices[hashedentry] == HashTableT::sentinel_v) return HashTableT::sentinel_v;
        hashedentry += 127;  ///< search next entry
        if (hashedentry > _tableSize) hashedentry = hashedentry % _tableSize;
      }
    }

    table_t _table;
    const value_t _tableSize;
    value_t *_cnt;
    key_t *_activeKeys;

  protected:
    constexpr value_t do_hash(const key_t &key) const {
      std::size_t ret = key[0];
      for (int d = 1; d < HashTableT::dim; ++d) hash_combine(ret, key[d]);
      return static_cast<value_t>(ret);
    }
  };

  template <execspace_e ExecSpace, typename Tn, int dim, typename Index>
  constexpr decltype(auto) proxy(HashTable<Tn, dim, Index> &table) {
    return HashTableView<ExecSpace, HashTable<Tn, dim, Index>>{table};
  }
  template <execspace_e ExecSpace, typename Tn, int dim, typename Index>
  constexpr decltype(auto) proxy(const HashTable<Tn, dim, Index> &table) {
    return HashTableView<ExecSpace, const HashTable<Tn, dim, Index>>{table};
  }

}  // namespace zs
