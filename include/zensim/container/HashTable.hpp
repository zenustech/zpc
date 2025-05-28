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

  template <typename Tn_, int dim_, typename Index = int, typename AllocatorT = ZSPmrAllocator<>>
  struct HashTable {
    static_assert(is_zs_allocator<AllocatorT>::value,
                  "Hashtable only works with zspmrallocator for now.");
    static_assert(is_same_v<Tn_, remove_cvref_t<Tn_>>, "Key is not cvref-unqualified type!");
    static_assert(std::is_default_constructible_v<Tn_>, "Key is not default-constructible!");
    static_assert(std::is_trivially_copyable_v<Tn_>, "Key is not trivially-copyable!");

    static constexpr int dim = dim_;
    using Tn = zs::make_signed_t<Tn_>;
    using key_t = vec<Tn, dim>;
    using value_t = zs::make_signed_t<Index>;
    using status_t = int;

    using index_type = Tn;

    using value_type = key_t;
    using allocator_type = AllocatorT;
    using size_type = zs::make_unsigned_t<value_t>;
    using difference_type = zs::make_signed_t<size_type>;
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
      Table(const allocator_type &allocator, size_t numEntries)
          : keys{allocator, numEntries},
            indices{allocator, numEntries},
            status{allocator, numEntries} {}
      void resize(size_type size) {
        keys.resize(size);
        indices.resize(size);
        status.resize(size);
      }

      Vector<key_t, allocator_type> keys;
      Vector<value_t, allocator_type> indices;
      Vector<status_t, allocator_type> status;
    };

    static_assert(
        std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_swap::value,
        "allocator should propagate on copy, move and swap (for impl simplicity)!");

    static constexpr Tn key_scalar_sentinel_v = detail::deduce_numeric_max<Tn>();
    static constexpr value_t sentinel_v{-1};  // this requires value_t to be signed type
    static constexpr status_t status_sentinel_v{-1};
    static constexpr size_t reserve_ratio_v = 16;

    constexpr decltype(auto) memoryLocation() const noexcept {
      return _cnt.get_allocator().location;
    }
    constexpr ProcID devid() const noexcept { return memoryLocation().devid(); }
    constexpr memsrc_e memspace() const noexcept { return memoryLocation().memspace(); }
    decltype(auto) get_allocator() const noexcept { return _cnt.get_allocator(); }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      if constexpr (is_virtual_zs_allocator<allocator_type>::value)
        return get_virtual_memory_source(mre, devid, (size_t)1 << (size_t)36, "STACK");
      else
        return get_memory_source(mre, devid);
    }

    constexpr auto &self() noexcept { return _table; }
    constexpr const auto &self() const noexcept { return _table; }

    constexpr size_t evaluateTableSize(size_t entryCnt) const {
      if (entryCnt == 0) return (size_t)0;
      return next_2pow(entryCnt) * reserve_ratio_v;
    }
    HashTable(const allocator_type &allocator, size_t numExpectedEntries)
        : _table{allocator, evaluateTableSize(numExpectedEntries)},
          _tableSize{static_cast<value_t>(evaluateTableSize(numExpectedEntries))},
          _cnt{allocator, 1},
          _activeKeys{allocator, evaluateTableSize(numExpectedEntries)} {
      _cnt.setVal((value_t)0);
    }
    HashTable(size_t numExpectedEntries, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : HashTable{get_default_allocator(mre, devid), numExpectedEntries} {}
    HashTable(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : HashTable{get_default_allocator(mre, devid), (size_t)0} {}

    ~HashTable() = default;

    HashTable(const HashTable &o)
        : _table{o._table}, _tableSize{o._tableSize}, _cnt{o._cnt}, _activeKeys{o._activeKeys} {}
    HashTable &operator=(const HashTable &o) {
      if (this == &o) return *this;
      HashTable tmp(o);
      swap(tmp);
      return *this;
    }
    HashTable clone(const allocator_type &allocator) const {
      HashTable ret{allocator, _tableSize / reserve_ratio_v};
      if (_cnt.size() > 0) ret._cnt.setVal(_cnt.getVal());
      ret._tableSize = _tableSize;
      ret._activeKeys = _activeKeys.clone(allocator);
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
      std::swap(_tableSize, o._tableSize);
      std::swap(_cnt, o._cnt);
      std::swap(_activeKeys, o._activeKeys);
    }
    friend void swap(HashTable &a, HashTable &b) noexcept { a.swap(b); }

    inline value_t size() const { return _cnt.getVal(0); }

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

    template <typename Policy> void resize(Policy &&, size_t numExpectedEntries);
    template <typename Policy> void preserve(Policy &&, size_t numExpectedEntries);
    template <typename Policy> void reset(Policy &&, bool clearCnt);

    Table _table;
    value_t _tableSize;
    Vector<value_t, allocator_type> _cnt;
    Vector<key_t, allocator_type> _activeKeys;
  };

#define ZS_FWD_DECL_HASHTABLE_INSTANTIATIONS(CoordIndexType, IndexType)                       \
  ZPC_FWD_DECL_TEMPLATE_STRUCT HashTable<CoordIndexType, 1, IndexType, ZSPmrAllocator<>>;     \
  ZPC_FWD_DECL_TEMPLATE_STRUCT HashTable<CoordIndexType, 2, IndexType, ZSPmrAllocator<>>;     \
  ZPC_FWD_DECL_TEMPLATE_STRUCT HashTable<CoordIndexType, 3, IndexType, ZSPmrAllocator<>>;     \
  ZPC_FWD_DECL_TEMPLATE_STRUCT HashTable<CoordIndexType, 4, IndexType, ZSPmrAllocator<>>;     \
  ZPC_FWD_DECL_TEMPLATE_STRUCT HashTable<CoordIndexType, 1, IndexType, ZSPmrAllocator<true>>; \
  ZPC_FWD_DECL_TEMPLATE_STRUCT HashTable<CoordIndexType, 2, IndexType, ZSPmrAllocator<true>>; \
  ZPC_FWD_DECL_TEMPLATE_STRUCT HashTable<CoordIndexType, 3, IndexType, ZSPmrAllocator<true>>; \
  ZPC_FWD_DECL_TEMPLATE_STRUCT HashTable<CoordIndexType, 4, IndexType, ZSPmrAllocator<true>>;

  ZS_FWD_DECL_HASHTABLE_INSTANTIATIONS(i32, i32)
  ZS_FWD_DECL_HASHTABLE_INSTANTIATIONS(i32, i64)

  template <typename HashTableView> struct ResetHashTable {
    using hash_table_type = typename HashTableView::hash_table_type;
    explicit ResetHashTable(HashTableView tv, bool clearCnt) : table{tv}, clearCnt{clearCnt} {}

    constexpr void operator()(typename HashTableView::size_type entry) noexcept {
      using namespace placeholders;
      table._table.keys[entry] = hash_table_type::key_t::constant(
          detail::deduce_numeric_max<typename hash_table_type::index_type>());
      table._table.indices[entry]
          = hash_table_type::sentinel_v;  // necessary for query to terminate
      table._table.status[entry] = -1;
      if (entry == 0 && clearCnt) *table._cnt = 0;
    }

    HashTableView table;
    bool clearCnt;
  };
  template <typename HashTableView> struct ReinsertHashTable {
    explicit ReinsertHashTable(HashTableView tv) : table{tv} {}

    constexpr void operator()(typename HashTableView::size_type entry) noexcept {
      table.insert(table._activeKeys[entry], entry);
    }

    HashTableView table;
  };
  template <typename HashTableView> struct RemoveHashTableEntries {
    using hash_table_type = typename HashTableView::hash_table_type;
    explicit RemoveHashTableEntries(HashTableView tv) : table{tv} {}
    constexpr void operator()(typename HashTableView::size_type blockno) noexcept {
      auto blockid = table._activeKeys[blockno];
      auto entry = table.entry(blockid);
      if (entry == hash_table_type::sentinel_v)
        printf("%llu-th key does not exist in the table??\n", (unsigned long long)blockno);
      table._table.keys[entry] = hash_table_type::key_t::constant(
          detail::deduce_numeric_max<typename hash_table_type::index_type>());
      table._table.indices[entry]
          = hash_table_type::sentinel_v;  // necessary for query to terminate
      table._table.status[entry] = -1;
    }
    HashTableView table;
  };

  template <typename Tn, int dim, typename Index, typename Allocator> template <typename Policy>
  void HashTable<Tn, dim, Index, Allocator>::preserve(Policy &&policy, size_t numExpectedEntries) {
    constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
    const auto numEntries = size();
    if (numExpectedEntries == numEntries) return;
    using LsvT = decltype(proxy<space>(*this));
    _cnt.setVal(numExpectedEntries);
    const auto newTableSize = evaluateTableSize(numExpectedEntries);
    _activeKeys.resize(newTableSize);  // newTableSize must be larger than numExpectedEntries
    /// clear table
    if (newTableSize > _tableSize) {
      _table.resize(newTableSize);
      _tableSize = newTableSize;
      policy(range(newTableSize),
             ResetHashTable<LsvT>{proxy<space>(*this), false});  // don't clear cnt
    } else
      policy(range(numEntries), RemoveHashTableEntries<LsvT>{proxy<space>(*this)});
    policy(range(std::min((size_t)numEntries, numExpectedEntries)),
           ReinsertHashTable<LsvT>{proxy<space>(*this)});
  }

  template <typename Tn, int dim, typename Index, typename Allocator> template <typename Policy>
  void HashTable<Tn, dim, Index, Allocator>::resize(Policy &&policy, size_t numExpectedEntries) {
    constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
    const auto newTableSize = evaluateTableSize(numExpectedEntries);
    if (newTableSize <= _tableSize) return;
    _table.resize(newTableSize);
    _tableSize = newTableSize;
    _activeKeys.resize(newTableSize);
    policy(range(newTableSize), ResetHashTable{proxy<space>(*this), false});  // don't clear cnt
    const auto numEntries = size();
    policy(range(numEntries), ReinsertHashTable{proxy<space>(*this)});
  }

  template <typename Tn, int dim, typename Index, typename Allocator> template <typename Policy>
  void HashTable<Tn, dim, Index, Allocator>::reset(Policy &&policy, bool clearCnt) {
    constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
    using LsvT = decltype(proxy<space>(*this));
    policy(range(_tableSize), ResetHashTable<LsvT>{proxy<space>(*this), clearCnt});
  }

#if 0
  using GeneralHashTable = variant<HashTable<i32, 2, int>, HashTable<i32, 2, long long int>,
                                   HashTable<i32, 3, int>, HashTable<i32, 3, long long int>>;
#else
  using GeneralHashTable = variant<HashTable<i32, 3, int>, HashTable<i32, 2, int>>;
#endif

  /// proxy to work within each backends
  template <execspace_e space, typename HashTableT, typename = void> struct HashTableView {
    static constexpr bool is_const_structure = is_const_v<HashTableT>;
    using hash_table_type = remove_const_t<HashTableT>;
    static constexpr int dim = hash_table_type::dim;
    static constexpr auto exectag = wrapv<space>{};
    using pointer = typename hash_table_type::pointer;
    using value_type = typename hash_table_type::value_type;
    using reference = typename hash_table_type::reference;
    using const_reference = typename hash_table_type::const_reference;
    using size_type = typename hash_table_type::size_type;
    using difference_type = typename hash_table_type::difference_type;
    using Tn = typename hash_table_type::Tn;
    using key_t = typename hash_table_type::key_t;
    using value_t = typename hash_table_type::value_t;
    using unsigned_value_t
        = conditional_t<sizeof(value_t) == 2, u16, conditional_t<sizeof(value_t) == 4, u32, u64>>;
    static_assert(sizeof(value_t) == sizeof(unsigned_value_t),
                  "value_t and unsigned_value_t of different sizes");
    using status_t = typename hash_table_type::status_t;
    struct table_t {
      conditional_t<is_const_structure, const key_t *, key_t *> keys{nullptr};
      conditional_t<is_const_structure, const value_t *, value_t *> indices{nullptr};
      conditional_t<is_const_structure, const status_t *, status_t *> status{nullptr};
    };

    static constexpr auto key_scalar_sentinel_v = hash_table_type::key_scalar_sentinel_v;
    static constexpr auto sentinel_v = hash_table_type::sentinel_v;
    static constexpr auto status_sentinel_v = hash_table_type::status_sentinel_v;

    HashTableView() noexcept = default;
    explicit constexpr HashTableView(HashTableT &table)
        : _table{table.self().keys.data(), table.self().indices.data(), table.self().status.data()},
          _activeKeys{table._activeKeys.data()},
          _tableSize{table._tableSize},
          _cnt{table._cnt.data()} {}

#if defined(__CUDACC__) || defined(__MUSACC__) || defined(__HIPCC__)
    template <
        typename VecT, execspace_e S = space, bool V = is_const_structure,
        enable_if_all<S == execspace_e::cuda || S == execspace_e::musa || S == execspace_e::rocm,
                      !V, VecT::dim == 1, VecT::extent == dim,
                      std::is_convertible_v<typename VecT::value_type, Tn>>
        = 0>
    __forceinline__ __device__ value_t insert(const VecInterface<VecT> &key) noexcept {
      using namespace placeholders;
      constexpr key_t key_sentinel_v = key_t::constant(HashTableT::key_scalar_sentinel_v);
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
    template <typename VecT, execspace_e S = space, bool V = is_const_structure,
              enable_if_all<is_host_execution<S>(), !V, VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, Tn>>
              = 0>
    inline value_t insert(const VecInterface<VecT> &key) {
      using namespace placeholders;
      constexpr key_t key_sentinel_v = key_t::constant(HashTableT::key_scalar_sentinel_v);
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

#if defined(__CUDACC__) || defined(__MUSACC__) || defined(__HIPCC__)
    template <
        typename VecT, execspace_e S = space, bool V = is_const_structure,
        enable_if_all<S == execspace_e::cuda || S == execspace_e::musa || S == execspace_e::rocm,
                      !V, VecT::dim == 1, VecT::extent == dim,
                      std::is_convertible_v<typename VecT::value_type, Tn>>
        = 0>
    __forceinline__ __device__ bool insert(const VecInterface<VecT> &key, value_t id) noexcept {
      using namespace placeholders;
      constexpr key_t key_sentinel_v = key_t::constant(HashTableT::key_scalar_sentinel_v);
      value_t hashedentry = (do_hash(key) % _tableSize + _tableSize) % _tableSize;
      key_t storedKey = atomicKeyCAS(&_table.status[hashedentry], &_table.keys[hashedentry], key);
      for (; !(storedKey == key_sentinel_v || storedKey == key);) {
        hashedentry = (hashedentry + 127) % _tableSize;
        storedKey = atomicKeyCAS(&_table.status[hashedentry], &_table.keys[hashedentry], key);
      }
      if (storedKey == key_sentinel_v) {
        _table.indices[hashedentry] = id;
        return true;
      }
      return false;
    }
#endif
    template <typename VecT, execspace_e S = space, bool V = is_const_structure,
              bool Assert = is_host_execution<S> && !V && VecT::dim == 1 && VecT::extent == dim
                             && std::is_convertible_v<typename VecT::value_type, Tn>,
              enable_if_t<Assert> = 0>
    inline bool insert(const VecInterface<VecT> &key, value_t id) {
      using namespace placeholders;
      constexpr key_t key_sentinel_v = key_t::constant(HashTableT::key_scalar_sentinel_v);
      value_t hashedentry = (do_hash(key) % _tableSize + _tableSize) % _tableSize;
      key_t storedKey = atomicKeyCAS(&_table.status[hashedentry], &_table.keys[hashedentry], key);
      for (; !(storedKey == key_sentinel_v || storedKey == key);) {
        hashedentry = (hashedentry + 127) % _tableSize;
        storedKey = atomicKeyCAS(&_table.status[hashedentry], &_table.keys[hashedentry], key);
      }
      if (storedKey == key_sentinel_v) {
        _table.indices[hashedentry] = id;
        return true;
      }
      return false;
    }

    /// make sure no one else is inserting in the same time!
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           std::is_convertible_v<typename VecT::value_type, Tn>>
                             = 0>
    constexpr value_t query(const VecInterface<VecT> &key) const noexcept {
      using namespace placeholders;
      value_t hashedentry = (do_hash(key) % _tableSize + _tableSize) % _tableSize;
      while (true) {
        if (key == _table.keys[hashedentry]) return _table.indices[hashedentry];
        if (_table.indices[hashedentry] == HashTableT::sentinel_v) return HashTableT::sentinel_v;
        hashedentry += 127;  ///< search next entry
        if (hashedentry > _tableSize) hashedentry = hashedentry % _tableSize;
      }
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           std::is_convertible_v<typename VecT::value_type, Tn>>
                             = 0>
    constexpr value_t entry(const VecInterface<VecT> &key) const noexcept {
      using namespace placeholders;
      value_t hashedentry = (do_hash(key) % _tableSize + _tableSize) % _tableSize;
      while (true) {
        if (key == _table.keys[hashedentry]) return hashedentry;
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
      constexpr key_t key_sentinel_v = key_t::constant(HashTableT::key_scalar_sentinel_v);
      for (value_t entry = 0; entry < _tableSize; ++entry) {
        _table.keys[entry] = key_sentinel_v;
        _table.indices[entry] = HashTableT::sentinel_v;
        _table.status[entry] = HashTableT::status_sentinel_v;
      }
    }

    constexpr auto size() const noexcept { return *_cnt; }

    table_t _table{};
    conditional_t<is_const_structure, const key_t *, key_t *> _activeKeys{nullptr};
    value_t _tableSize{0};  // constness makes non-trivial
    conditional_t<is_const_structure, const value_t *, value_t *> _cnt{nullptr};

  protected:
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim,
                                           std::is_convertible_v<typename VecT::value_type, Tn>>
                             = 0>
    constexpr value_t do_hash(const VecInterface<VecT> &key) const noexcept {
      size_t ret = key[0];
      for (int d = 1; d < HashTableT::dim; ++d) hash_combine(ret, key[d]);
      return static_cast<value_t>(ret);
    }
#if defined(__CUDACC__) || defined(__MUSACC__) || defined(__HIPCC__)
    template <
        typename VecT, execspace_e S = space, bool V = is_const_structure,
        enable_if_all<S == execspace_e::cuda || S == execspace_e::musa || S == execspace_e::rocm,
                      !V, VecT::dim == 1, VecT::extent == dim,
                      std::is_convertible_v<typename VecT::value_type, Tn>>
        = 0>
    __forceinline__ __device__ key_t atomicKeyCAS(status_t *lock, volatile key_t *const dest,
                                                  const VecInterface<VecT> &val) noexcept {
      constexpr auto execTag = wrapv<S>{};
      using namespace placeholders;
      constexpr key_t key_sentinel_v = key_t::constant(HashTableT::key_scalar_sentinel_v);
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
    template <typename VecT, execspace_e S = space, bool V = is_const_structure,
              enable_if_all<is_host_execution<S>(), !V, VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, Tn>>
              = 0>
    inline key_t atomicKeyCAS(status_t *lock, volatile key_t *const dest,
                              const VecInterface<VecT> &val) {
      constexpr auto execTag = wrapv<S>{};
      using namespace placeholders;
      constexpr key_t key_sentinel_v = key_t::constant(HashTableT::key_scalar_sentinel_v);
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
  decltype(auto) proxy(HashTable<Tn, dim, Index, Allocator> &table) {
    return HashTableView<ExecSpace, HashTable<Tn, dim, Index, Allocator>>{table};
  }
  template <execspace_e ExecSpace, typename Tn, int dim, typename Index, typename Allocator>
  decltype(auto) proxy(const HashTable<Tn, dim, Index, Allocator> &table) {
    return HashTableView<ExecSpace, const HashTable<Tn, dim, Index, Allocator>>{table};
  }

#if ZS_ENABLE_SERIALIZATION
  template <typename S, typename Tn, int dim, typename Index>
  void serialize(S &s, HashTable<Tn, dim, Index, ZSPmrAllocator<>> &ht) {
    if (!ht.memoryLocation().onHost()) {
      ht = ht.clone({memsrc_e::host, -1});
    }

    serialize(s, ht._table.keys);
    serialize(s, ht._table.indices);
    serialize(s, ht._table.status);
    s.template value<sizeof(ht._tableSize)>(ht._tableSize);
    serialize(s, ht._cnt);
    serialize(s, ht._activeKeys);
  }
#endif

}  // namespace zs
