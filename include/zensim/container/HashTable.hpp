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

  template <typename Tn_, int dim_, typename Index, typename AllocatorT = ZSPmrAllocator<>>
  struct HashTable {
    static_assert(is_zs_allocator<AllocatorT>::value,
                  "Hashtable only works with zspmrallocator for now.");
    static_assert(is_same_v<Tn_, remove_cvref_t<Tn_>>, "Key is not cvref-unqualified type!");
    static_assert(std::is_default_constructible_v<Tn_>, "Key is not default-constructible!");
    static_assert(std::is_trivially_copyable_v<Tn_>, "Key is not trivially-copyable!");

    static constexpr int dim = dim_;
    using Tn = std::make_signed_t<Tn_>;
    using key_t = vec<Tn, dim>;
    using value_t = std::make_signed_t<Index>;
    using status_t = int;

    using value_type = key_t;
    using allocator_type = AllocatorT;
    using size_type = std::make_unsigned_t<value_t>;
    using difference_type = std::make_signed_t<size_type>;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;

    struct Table {
      Table() = default;
      Table(const Table &) = default;
      Table(Table &&) noexcept = default;
      Table &operator=(const Table &) = default;
      Table &operator=(Table &&) noexcept = default;
      Table(const allocator_type &allocator, std::size_t numEntries)
          : keys{allocator, numEntries},
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
    decltype(auto) get_allocator() const noexcept { return _allocator; }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      if constexpr (is_virtual_zs_allocator<allocator_type>::value)
        return get_virtual_memory_source(mre, devid, (std::size_t)1 << (std::size_t)36, "STACK");
      else
        return get_memory_source(mre, devid);
    }

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
      value_t res[1];
      res[0] = (value_t)0;
      copy(MemoryEntity{_cnt.memoryLocation(), (void *)_cnt.data()},
           MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)res}, sizeof(value_t));
    }
    HashTable(std::size_t tableSize, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : HashTable{get_default_allocator(mre, devid), tableSize} {}
    HashTable(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : HashTable{get_default_allocator(mre, devid), (std::size_t)0} {}

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
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
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
      value_t res[1];
      copy(MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)res},
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

    template <typename Policy> void resize(Policy &&, std::size_t tableSize);

    Table _table;
    allocator_type _allocator;
    value_t _tableSize;
    Vector<value_t> _cnt;
    Vector<key_t> _activeKeys;
  };

  template <typename Tn, int dim, typename Index, typename Allocator> template <typename Policy>
  void HashTable<Tn, dim, Index, Allocator>::resize(Policy &&policy, std::size_t tableSize) {
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
#if 0
    const auto s = size();
    auto tags = getPropertyTags();
    tags.insert(std::end(tags), std::begin(appendTags), std::end(appendTags));
    HashTable<Tn, dim, Index, Allocator> tmp{get_allocator(), tableSize};
    policy(range(s), TileVectorCopy{proxy<space>(*this), proxy<space>(tmp)});
    *this = std::move(tmp);
#endif
  }

#if 0
  using GeneralHashTable = variant<HashTable<i32, 2, int>, HashTable<i32, 2, long long int>,
                                   HashTable<i32, 3, int>, HashTable<i32, 3, long long int>>;
#else
  using GeneralHashTable = variant<HashTable<i32, 3, int>, HashTable<i32, 2, int>>;
#endif

  /// proxy to work within each backends
  template <execspace_e space, typename HashTableT, typename = void> struct HashTableView {
    static constexpr bool is_const_structure = std::is_const_v<HashTableT>;
    using hash_table_type = std::remove_const_t<HashTableT>;
    static constexpr int dim = hash_table_type::dim;
    static constexpr auto exectag = wrapv<space>{};
    using Tn = typename hash_table_type::Tn;
    using key_t = typename hash_table_type::key_t;
    using value_t = typename hash_table_type::value_t;
    using unsigned_value_t = std::make_unsigned_t<value_t>;
    using status_t = typename hash_table_type::status_t;
    struct table_t {
      key_t *keys;
      value_t *indices;
      status_t *status;
    };

    constexpr HashTableView() = default;
    ~HashTableView() = default;

    explicit constexpr HashTableView(HashTableT &table)
        : _table{table.self().keys.data(), table.self().indices.data(), table.self().status.data()},
          _tableSize{table._tableSize},
          _cnt{table._cnt.data()},
          _activeKeys{table._activeKeys.data()} {}

#if defined(__CUDACC__)
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_t<S == execspace_e::cuda && !V> = 0>
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
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_t<S != execspace_e::cuda && !V> = 0>
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
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_t<S == execspace_e::host && !V> = 0>
    void clear() {
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
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_t<S == execspace_e::cuda && !V> = 0>
    __forceinline__ __device__ key_t atomicKeyCAS(status_t *lock, volatile key_t *const dest,
                                                  const key_t &val) {
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
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_t<S != execspace_e::cuda && !V> = 0>
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

  template <execspace_e ExecSpace, typename Tn, int dim, typename Index, typename Allocator>
  constexpr decltype(auto) proxy(HashTable<Tn, dim, Index, Allocator> &table) {
    return HashTableView<ExecSpace, HashTable<Tn, dim, Index, Allocator>>{table};
  }
  template <execspace_e ExecSpace, typename Tn, int dim, typename Index, typename Allocator>
  constexpr decltype(auto) proxy(const HashTable<Tn, dim, Index, Allocator> &table) {
    return HashTableView<ExecSpace, const HashTable<Tn, dim, Index, Allocator>>{table};
  }

}  // namespace zs
