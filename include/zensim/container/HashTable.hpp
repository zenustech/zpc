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
#include "zensim/types/RuntimeStructurals.hpp"

namespace zs {

  template <typename Key, typename Index, typename Status> using hash_table_snode
      = ds::snode_t<ds::decorations<ds::soa>, ds::uniform_domain<0, Index, 1, index_seq<0>>,
                    zs::tuple<wrapt<Key>, wrapt<Index>, Status>, vseq_t<1, 1, 1>>;

  template <typename Key, typename Index, typename Status = int> using hash_table_instance
      = ds::instance_t<ds::dense, hash_table_snode<Key, Index, Status>>;

  template <typename Tn_, int dim_, typename Index> struct HashTable : MemoryHandle {
    static_assert(is_same_v<Tn_, remove_cvref_t<Tn_>>, "Key is not cvref-unqualified type!");
    static_assert(std::is_default_constructible_v<Tn_>, "Key is not default-constructible!");
    static_assert(std::is_trivially_copyable_v<Tn_>, "Key is not trivially-copyable!");

    static constexpr int dim = dim_;
    using Tn = std::make_signed_t<Tn_>;
    using key_t = vec<Tn, dim>;
    using value_t = std::make_signed_t<Index>;
    using status_t = int;
    using base_t = hash_table_instance<key_t, value_t, status_t>;

    using value_type = key_t;
    using allocator_type = ZSPmrAllocator<>;
    using size_type = std::make_unsigned_t<value_t>;
    using difference_type = std::make_signed_t<size_type>;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;

    static_assert(
        std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_swap::value,
        "allocator should propagate on copy, move and swap (for impl simplicity)!");

    static constexpr Tn key_scalar_sentinel_v = std::numeric_limits<Tn>::max();
    static constexpr value_t sentinel_v{-1};  // this requires value_t to be signed type
    static constexpr status_t status_sentinel_v{-1};
    static constexpr std::size_t reserve_ratio_v = 16;

    constexpr MemoryHandle &base() noexcept { return static_cast<MemoryHandle &>(*this); }
    constexpr const MemoryHandle &base() const noexcept {
      return static_cast<const MemoryHandle &>(*this);
    }
    constexpr auto &self() noexcept { return _inst; }
    constexpr const auto &self() const noexcept { return _inst; }

    HashTable(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : MemoryHandle{mre, devid},
          _allocator{get_memory_source(mre, devid)},
          _tableSize{0},
          _cnt{mre, devid},
          _activeKeys{mre, devid} {
      _inst = buildInstance(mre, devid, _tableSize);
    }

    constexpr std::size_t evaluateTableSize(std::size_t entryCnt) const {
      return next_2pow(entryCnt) * reserve_ratio_v;
    }
    HashTable(std::size_t tableSize, memsrc_e mre = memsrc_e::host, ProcID devid = -1,
              std::size_t alignment = 0)
        : MemoryHandle{mre, devid},
          _allocator{get_memory_source(mre, devid)},
          _tableSize{static_cast<value_t>(evaluateTableSize(tableSize))},
          _cnt{1, mre, devid},
          _activeKeys{evaluateTableSize(tableSize), mre, devid} {
      _inst = buildInstance(mre, devid, _tableSize);
    }
    HashTable(mr_t *mr, std::size_t tableSize, memsrc_e mre = memsrc_e::host, ProcID devid = -1,
              std::size_t alignment = 0)
        : MemoryHandle{mre, devid},
          _allocator{mr},
          _tableSize{static_cast<value_t>(evaluateTableSize(tableSize))},
          _cnt{1, mre, devid},
          _activeKeys{evaluateTableSize(tableSize), mre, devid} {
      _inst = buildInstance(mre, devid, _tableSize);
    }
    HashTable(const SharedHolder<mr_t> &mr, std::size_t tableSize, memsrc_e mre = memsrc_e::host,
              ProcID devid = -1, std::size_t alignment = 0)
        : MemoryHandle{mre, devid},
          _allocator{mr},
          _tableSize{static_cast<value_t>(evaluateTableSize(tableSize))},
          _cnt{1, mre, devid},
          _activeKeys{evaluateTableSize(tableSize), mre, devid} {
      _inst = buildInstance(mre, devid, _tableSize);
    }
    HashTable(const allocator_type &allocator, std::size_t tableSize, memsrc_e mre = memsrc_e::host,
              ProcID devid = -1, std::size_t alignment = 0)
        : MemoryHandle{mre, devid},
          _allocator{allocator},
          _tableSize{static_cast<value_t>(evaluateTableSize(tableSize))},
          _cnt{1, mre, devid},
          _activeKeys{evaluateTableSize(tableSize), mre, devid} {
      _inst = buildInstance(mre, devid, _tableSize);
    }

    ~HashTable() {
      if (self().address() && self().node().extent() > 0) self().dealloc(_allocator);
    }

    HashTable(const HashTable &o)
        : MemoryHandle{o.base()},
          _allocator{o._allocator},
          _tableSize{o._tableSize},
          _cnt{o._cnt},
          _activeKeys{o._activeKeys} {
      _inst = buildInstance(this->memspace(), this->devid(), this->_tableSize);
      if (ds::snode_size(o.self().template node<0>()) > 0)
        copy(MemoryEntity{base(), (void *)self().address()},
             MemoryEntity{o.base(), (void *)o.self().address()},
             ds::snode_size(o.self().template node<0>()));
    }
    HashTable &operator=(const HashTable &o) {
      if (this == &o) return *this;
      HashTable tmp(o);
      swap(tmp);
      return *this;
    }
    HashTable clone(const MemoryHandle &mh, const allocator_type &allocator) const {
      HashTable ret{allocator, _tableSize / reserve_ratio_v, mh.memspace(), mh.devid()};
      if (_cnt.size() > 0)
        copy(MemoryEntity{ret._cnt.base(), ret._cnt.data()}, MemoryEntity{_cnt.base(), _cnt.data()},
             sizeof(value_t));
      if (_activeKeys.size() > 0)
        copy(MemoryEntity{ret._activeKeys.base(), ret._activeKeys.data()},
             MemoryEntity{_activeKeys.base(), _activeKeys.data()},
             sizeof(key_t) * _activeKeys.size());
      if (ds::snode_size(self().template node<0>()) > 0)
        copy(MemoryEntity{ret.base(), (void *)ret.self().address()},
             MemoryEntity{base(), (void *)self().address()},
             ds::snode_size(self().template node<0>()));
      return ret;
    }
    HashTable clone(const MemoryHandle &mh) const {
      return clone(mh, get_memory_source(mh.memspace(), mh.devid()));
    }

    HashTable(HashTable &&o) noexcept {
      const HashTable defaultTable{};
      base() = std::exchange(o.base(), defaultTable.base());
      self() = std::exchange(o.self(), defaultTable.self());
      _allocator = std::exchange(o._allocator, defaultTable._allocator);
      _tableSize = std::exchange(o._tableSize, defaultTable._tableSize);
      /// critical! use user-defined move assignment constructor!
      _cnt = std::move(o._cnt);
      _activeKeys = std::move(o._activeKeys);
    }
    HashTable &operator=(HashTable &&o) noexcept {
      if (this == &o) return *this;
      HashTable tmp(std::move(o));
      swap(tmp);
      return *this;
    }
    void swap(HashTable &o) noexcept {
      std::swap(base(), o.base());
      std::swap(self(), o.self());
      std::swap(_allocator, o._allocator);
      std::swap(_tableSize, o._tableSize);
      std::swap(_cnt, o._cnt);
      std::swap(_activeKeys, o._activeKeys);
    }
    friend void swap(HashTable &a, HashTable &b) { a.swap(b); }

    inline value_t size() const {
      Vector<value_t> res{1, memsrc_e::host, -1};
      copy(MemoryEntity{res.base(), (void *)res.data()},
           MemoryEntity{_cnt.base(), (void *)_cnt.data()}, sizeof(value_t));
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

    base_t _inst;
    allocator_type _allocator;
    value_t _tableSize;
    Vector<value_t> _cnt;
    Vector<key_t> _activeKeys;

  protected:
    constexpr auto buildInstance(memsrc_e mre, ProcID devid, value_t capacity) {
      using namespace ds;
      uniform_domain<0, value_t, 1, index_seq<0>> dom{wrapv<0>{}, capacity};
      hash_table_snode<key_t, value_t, status_t> node{
          ds::decorations<ds::soa>{}, dom,
          zs::make_tuple(wrapt<key_t>{}, wrapt<value_t>{}, wrapt<status_t>{}), vseq_t<1, 1, 1>{}};
      auto inst = instance{wrapv<dense>{}, zs::make_tuple(node)};
      if (capacity) inst.alloc(_allocator);
      return inst;
    }
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
    using table_t = typename HashTableT::base_t;
    using key_t = typename HashTableT::key_t;
    using value_t = typename HashTableT::value_t;
    using unsigned_value_t = std::make_unsigned_t<value_t>;
    using status_t = typename HashTableT::status_t;

    constexpr HashTableView() = default;
    ~HashTableView() = default;

    explicit constexpr HashTableView(HashTableT &table)
        : _table{table.self()},
          _tableSize{table._tableSize},
          _cnt{table._cnt.data()},
          _activeKeys{table._activeKeys.data()} {}

    ZS_FUNCTION value_t insert(const key_t &key) {
      using namespace placeholders;
      constexpr key_t key_sentinel_v = key_t::uniform(HashTableT::key_scalar_sentinel_v);
      value_t hashedentry = (do_hash(key) % _tableSize + _tableSize) % _tableSize;
      key_t storedKey = atomicKeyCAS(&_table(_2, hashedentry), &_table(_0, hashedentry), key);
      for (; !(storedKey == key_sentinel_v || storedKey == key);) {
        hashedentry = (hashedentry + 127) % _tableSize;
        storedKey = atomicKeyCAS(&_table(_2, hashedentry), &_table(_0, hashedentry), key);
      }
      if (storedKey == key_sentinel_v) {
        auto localno = (value_t)atomic_add(exectag, (unsigned_value_t *)_cnt, (unsigned_value_t)1);
        _table(_1, hashedentry) = localno;
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
        if (key == (key_t)_table(_0, hashedentry)) return _table(_1, hashedentry);
        if (_table(_1, hashedentry) == HashTableT::sentinel_v) return HashTableT::sentinel_v;
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
        _table(_0, entry) = key_sentinel_v;
        _table(_1, entry) = HashTableT::sentinel_v;
        _table(_2, entry) = HashTableT::status_sentinel_v;
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
    template <execspace_e S = space, enable_if_t<S == execspace_e::cuda> = 0>
    ZS_FUNCTION key_t atomicKeyCAS(status_t *lock, volatile key_t *const dest, const key_t &val) {
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
    template <execspace_e S = space, enable_if_t<S != execspace_e::cuda> = 0>
    ZS_FUNCTION key_t atomicKeyCAS(status_t *lock, volatile key_t *const dest, const key_t &val) {
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

  template <execspace_e ExecSpace, typename Tn, int dim, typename Index>
  constexpr decltype(auto) proxy(HashTable<Tn, dim, Index> &table) {
    return HashTableView<ExecSpace, HashTable<Tn, dim, Index>>{table};
  }

}  // namespace zs
