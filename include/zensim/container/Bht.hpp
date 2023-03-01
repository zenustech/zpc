#pragma once
#include "zensim/container/Vector.hpp"
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

  template <typename Tn_, int dim_, typename Index = int, int B = 32,
            typename AllocatorT = ZSPmrAllocator<>>
  struct bht {
    static_assert(is_zs_allocator<AllocatorT>::value,
                  "Hashtable only works with zspmrallocator for now.");
    static_assert(is_same_v<Tn_, remove_cvref_t<Tn_>>, "Key is not cvref-unqualified type!");
    static_assert(std::is_default_constructible_v<Tn_>, "Key is not default-constructible!");
    static_assert(std::is_trivially_copyable_v<Tn_>, "Key is not trivially-copyable!");
    static_assert(std::is_fundamental_v<Tn_>, "Key component should be fundamental!");

    using index_type = std::make_signed_t<Tn_>;
    using key_t = vec<index_type, dim_>;
    using key_type = vec<index_type, dim_>;
    using value_t = std::make_signed_t<Index>;
    using status_t = int;
    using status_type = int;
    using size_type = std::make_unsigned_t<Index>;

    static constexpr int dim = dim_;
    static constexpr size_type bucket_size = B;
    static constexpr size_type threshold = bucket_size - 2;
    static_assert(threshold > 0, "bucket load threshold must be positive");

    struct alignas(next_2pow(sizeof(index_type) * dim)) storage_key_type {
      static constexpr std::size_t num_total_bytes = next_2pow(sizeof(index_type) * dim);
      static constexpr std::size_t num_padded_bytes
          = next_2pow(sizeof(index_type) * dim) - sizeof(index_type) * dim;
      constexpr storage_key_type() noexcept = default;
      constexpr storage_key_type(const key_t &k) noexcept : val{k} {}
      constexpr storage_key_type(key_t &&k) noexcept : val{std::move(k)} {}
      ~storage_key_type() = default;
      constexpr operator key_t &() { return val; }
      constexpr operator const key_t &() const { return val; }
      constexpr operator volatile key_t &() volatile { return val; }
      constexpr operator const volatile key_t &() const volatile { return val; }
      key_t val;
    };

    using value_type = key_t;
    using allocator_type = AllocatorT;
    using difference_type = std::make_signed_t<size_type>;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;

    using hasher_type = universal_hash<key_t>;
    using pair_type = zs::tuple<key_t, value_t>;

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
      void resize(size_type size) {
        keys.resize(size);
        indices.resize(size);
        status.resize(size);
      }

      Vector<storage_key_type, allocator_type> keys;
      Vector<value_t, allocator_type> indices;
      Vector<status_t, allocator_type> status;
    };

    static_assert(
        std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_swap::value,
        "allocator should propagate on copy, move and swap (for impl simplicity)!");

    static constexpr index_type key_scalar_sentinel_v = limits<index_type>::max();
    static constexpr index_type sentinel_v{-1};  // this requires value_t to be signed type
    static constexpr status_t status_sentinel_v{-1};

    constexpr decltype(auto) memoryLocation() const noexcept {
      return _cnt.get_allocator().location;
    }
    constexpr ProcID devid() const noexcept { return memoryLocation().devid(); }
    constexpr memsrc_e memspace() const noexcept { return memoryLocation().memspace(); }
    decltype(auto) get_allocator() const noexcept { return _cnt.get_allocator(); }
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
      std::size_t n = next_2pow(entryCnt) * 2;
      return n + (bucket_size - n % bucket_size);
    }
    bht(const allocator_type &allocator, std::size_t numExpectedEntries)
        : _table{allocator, evaluateTableSize(numExpectedEntries)},
          _tableSize{static_cast<value_t>(evaluateTableSize(numExpectedEntries))},
          _cnt{allocator, 1},
          _activeKeys{allocator, evaluateTableSize(numExpectedEntries)},
          _buildSuccess{allocator, 1} {
      std::mt19937 rng(2);
      // initialize hash funcs
      _hf0 = hasher_type(rng);
      _hf1 = hasher_type(rng);
      _hf2 = hasher_type(rng);

      _buildSuccess.setVal((value_t)1);

      _cnt.setVal((value_t)0);
    }
    bht(std::size_t numExpectedEntries, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : bht{get_default_allocator(mre, devid), numExpectedEntries} {}
    bht(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : bht{get_default_allocator(mre, devid), (std::size_t)0} {}

    ~bht() = default;

    bht(const bht &o)
        : _table{o._table},
          _tableSize{o._tableSize},
          _cnt{o._cnt},
          _activeKeys{o._activeKeys},
          _buildSuccess{o._buildSuccess},
          _hf0{o._hf0},
          _hf1{o._hf1},
          _hf2{o._hf2} {}
    bht &operator=(const bht &o) {
      if (this == &o) return *this;
      bht tmp(o);
      swap(tmp);
      return *this;
    }
    bht clone(const allocator_type &allocator) const {
      bht ret{allocator, _tableSize / bucket_size};
      if (_cnt.size() > 0) ret._cnt.setVal(_cnt.getVal());
      ret._tableSize = _tableSize;
      ret._activeKeys = _activeKeys.clone(allocator);
      ret._buildSuccess = _buildSuccess.clone(allocator);
      ret._hf0 = _hf0;
      ret._hf1 = _hf1;
      ret._hf2 = _hf2;
      if (_tableSize > 0) {
        ret.self().keys = self().keys.clone(allocator);
        ret.self().indices = self().indices.clone(allocator);
        ret.self().status = self().status.clone(allocator);
      }
      return ret;
    }
    bht clone(const MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }

    bht(bht &&o) noexcept {
      const bht defaultTable{};
      self() = std::exchange(o.self(), defaultTable.self());
      _tableSize = std::exchange(o._tableSize, defaultTable._tableSize);
      /// critical! use user-defined move assignment constructor!
      _cnt = std::exchange(o._cnt, defaultTable._cnt);
      _activeKeys = std::exchange(o._activeKeys, defaultTable._activeKeys);
      _buildSuccess = std::exchange(o._buildSuccess, defaultTable._buildSuccess);
      _hf0 = std::exchange(o._hf0, defaultTable._hf0);
      _hf1 = std::exchange(o._hf1, defaultTable._hf1);
      _hf2 = std::exchange(o._hf2, defaultTable._hf2);
    }
    bht &operator=(bht &&o) noexcept {
      if (this == &o) return *this;
      bht tmp(std::move(o));
      swap(tmp);
      return *this;
    }
    void swap(bht &o) noexcept {
      std::swap(self(), o.self());
      std::swap(_tableSize, o._tableSize);
      std::swap(_cnt, o._cnt);
      std::swap(_activeKeys, o._activeKeys);
      std::swap(_buildSuccess, o._buildSuccess);
      std::swap(_hf0, o._hf0);
      std::swap(_hf1, o._hf1);
      std::swap(_hf2, o._hf2);
    }
    friend void swap(bht &a, bht &b) noexcept { a.swap(b); }

    inline value_t size() const { return _cnt.getVal(0); }

#if 0
    template <typename Policy>
    void resize(Policy &&, std::size_t numExpectedEntries);
    template <typename Policy>
    void preserve(Policy &&, std::size_t numExpectedEntries);
#endif
    template <typename Policy> void reset(Policy &&, bool clearCnt);

    Table _table;
    value_t _tableSize;
    Vector<value_t, allocator_type> _cnt;
    Vector<key_t, allocator_type> _activeKeys;
    Vector<int, allocator_type> _buildSuccess;
    hasher_type _hf0, _hf1, _hf2;
  };

#define EXTERN_BHT_INSTANTIATIONS(CoordIndexType, IndexType, B)                      \
  extern template struct bht<CoordIndexType, 1, IndexType, B, ZSPmrAllocator<>>;     \
  extern template struct bht<CoordIndexType, 2, IndexType, B, ZSPmrAllocator<>>;     \
  extern template struct bht<CoordIndexType, 3, IndexType, B, ZSPmrAllocator<>>;     \
  extern template struct bht<CoordIndexType, 4, IndexType, B, ZSPmrAllocator<>>;     \
  extern template struct bht<CoordIndexType, 1, IndexType, B, ZSPmrAllocator<true>>; \
  extern template struct bht<CoordIndexType, 2, IndexType, B, ZSPmrAllocator<true>>; \
  extern template struct bht<CoordIndexType, 3, IndexType, B, ZSPmrAllocator<true>>; \
  extern template struct bht<CoordIndexType, 4, IndexType, B, ZSPmrAllocator<true>>;

  EXTERN_BHT_INSTANTIATIONS(i32, i32, 16)
  EXTERN_BHT_INSTANTIATIONS(i32, i64, 16)
  EXTERN_BHT_INSTANTIATIONS(i64, i64, 16)

  template <typename HashTableView> struct ResetBHT {
    using hash_table_type = typename HashTableView::hash_table_type;
    explicit ResetBHT(HashTableView tv, bool clearCnt) : table{tv}, clearCnt{clearCnt} {}

    constexpr void operator()(typename HashTableView::size_type entry) noexcept {
      using namespace placeholders;
      table._table.keys[entry]
          = hash_table_type::key_t::constant(hash_table_type::key_scalar_sentinel_v);
      table._table.indices[entry]
          = hash_table_type::sentinel_v;  // necessary for query to terminate
      table._table.status[entry] = -1;
      if (entry == 0 && clearCnt) *table._cnt = 0;
    }

    HashTableView table;
    bool clearCnt;
  };

  template <typename Tn, int dim, typename Index, int B, typename Allocator>
  template <typename Policy>
  void bht<Tn, dim, Index, B, Allocator>::reset(Policy &&policy, bool clearCnt) {
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    using LsvT = decltype(proxy<space>(*this));
    policy(range(_tableSize), ResetBHT<LsvT>{proxy<space>(*this), clearCnt});
  }

  /// proxy to work within each backends
  template <execspace_e space, typename HashTableT, typename = void> struct BHTView {
    static constexpr bool is_const_structure = std::is_const_v<HashTableT>;
    using hash_table_type = std::remove_const_t<HashTableT>;
    static constexpr auto exectag = wrapv<space>{};
    using pointer = typename hash_table_type::pointer;
    using value_type = typename hash_table_type::value_type;
    using reference = typename hash_table_type::reference;
    using const_reference = typename hash_table_type::const_reference;
    using size_type = typename hash_table_type::size_type;
    using difference_type = typename hash_table_type::difference_type;
    using Tn = typename hash_table_type::index_type;
    using index_type = typename hash_table_type::index_type;
    using storage_key_type = typename hash_table_type::storage_key_type;
    using key_t = typename hash_table_type::key_t;
    using value_t = typename hash_table_type::value_t;
    using unsigned_value_t
        = conditional_t<sizeof(value_t) == 2, u16, conditional_t<sizeof(value_t) == 4, u32, u64>>;
    using status_t = typename hash_table_type::status_t;
    static_assert(sizeof(value_t) == sizeof(unsigned_value_t),
                  "value_t and unsigned_value_t of different sizes");

    static constexpr int dim = hash_table_type::dim;
    static constexpr size_type bucket_size = hash_table_type::bucket_size;
    static constexpr size_type threshold = hash_table_type::threshold;

    using hasher_type = typename hash_table_type::hasher_type;

    struct table_t {
      conditional_t<is_const_structure, const storage_key_type *, storage_key_type *> keys{nullptr};
      conditional_t<is_const_structure, const value_t *, value_t *> indices{nullptr};
      conditional_t<is_const_structure, const status_t *, status_t *> status{nullptr};
    };

    static constexpr auto key_scalar_sentinel_v = hash_table_type::key_scalar_sentinel_v;
    static constexpr auto sentinel_v = hash_table_type::sentinel_v;
    static constexpr auto status_sentinel_v = hash_table_type::status_sentinel_v;

    BHTView() noexcept = default;
    explicit constexpr BHTView(HashTableT &table)
        : _table{table.self().keys.data(), table.self().indices.data(), table.self().status.data()},
          _activeKeys{table._activeKeys.data()},
          _tableSize{table._tableSize},
          _cnt{table._cnt.data()},
          _success{table._buildSuccess.data()},
          _numBuckets{table._tableSize / bucket_size},
          _hf0{table._hf0},
          _hf1{table._hf1},
          _hf2{table._hf2} {}

#if defined(__CUDACC__) && ZS_ENABLE_CUDA
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_all<S == execspace_e::cuda, !V> = 0>
    __forceinline__ __host__ __device__ value_t insert(const key_t &key) noexcept {
      constexpr key_t key_sentinel_v = key_t::constant(HashTableT::key_scalar_sentinel_v);

      int iter = 0;
      int load = 0;
      size_type bucketOffset = _hf0(key) % _numBuckets * bucket_size;
      while (iter < 3) {
        for (; load != bucket_size; ++load) {
          key_t curKey = atomicLoad(
              &_table.status[bucketOffset + load],
              const_cast<const volatile storage_key_type *>(&_table.keys[bucketOffset + load]));
          if (curKey == key) {
            load = -1;
            break;
          } else if (curKey == key_sentinel_v)
            break;
        }
        if (load < 0)
          return HashTableT::sentinel_v;
        else if (load <= threshold) {
          key_t storedKey = key_sentinel_v;
          if (atomicSwitchIfEqual(
                  &_table.status[bucketOffset + load],
                  const_cast<volatile storage_key_type *>(&_table.keys[bucketOffset + load]),
                  key)) {
            auto localno
                = (value_t)atomic_add(exectag, (unsigned_value_t *)_cnt, (unsigned_value_t)1);
            _table.indices[bucketOffset + load] = localno;
            _activeKeys[localno] = key;
            if (localno >= _tableSize - 20)
              printf("proximity!!! %d -> %d\n", (int)localno, (int)_tableSize);
            return localno;  ///< only the one that inserts returns the actual index
          }
        } else {
          ++iter;
          load = 0;
          if (iter == 1)
            bucketOffset = _hf1(key) % _numBuckets * bucket_size;
          else if (iter == 2)
            bucketOffset = _hf2(key) % _numBuckets * bucket_size;
          else
            break;
        }
      }
      *_success = false;
      return HashTableT::sentinel_v;
    }
#endif
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_all<S != execspace_e::cuda, !V> = 0>
    inline value_t insert(const key_t &key) noexcept {
      constexpr key_t key_sentinel_v = key_t::constant(HashTableT::key_scalar_sentinel_v);
      constexpr value_t sentinel_v = HashTableT::sentinel_v;

      int iter = 0;
      int load = 0;
      size_type bucketOffset = _hf0(key) % _numBuckets * bucket_size;
      while (iter < 3) {
        for (; load != bucket_size; ++load) {
          key_t curKey = atomicLoad(
              &_table.status[bucketOffset + load],
              const_cast<const volatile storage_key_type *>(&_table.keys[bucketOffset + load]));
          if (curKey == key) {
            load = -1;
            break;
          } else if (curKey == key_sentinel_v)
            break;
        }
        if (load < 0)
          return sentinel_v;
        else if (load <= threshold) {
          key_t storedKey = key_sentinel_v;
          if (atomicSwitchIfEqual(
                  &_table.status[bucketOffset + load],
                  const_cast<volatile storage_key_type *>(&_table.keys[bucketOffset + load]),
                  key)) {
            auto localno
                = (value_t)atomic_add(exectag, (unsigned_value_t *)_cnt, (unsigned_value_t)1);
            _table.indices[bucketOffset + load] = localno;
            _activeKeys[localno] = key;
            if (localno >= _tableSize - 20)
              printf("proximity!!! %d -> %d\n", (int)localno, (int)_tableSize);
            return localno;  ///< only the one that inserts returns the actual index
          }
        } else {
          ++iter;
          load = 0;
          if (iter == 1)
            bucketOffset = _hf1(key) % _numBuckets * bucket_size;
          else if (iter == 2)
            bucketOffset = _hf2(key) % _numBuckets * bucket_size;
          else
            break;
        }
      }
      *_success = false;
      return HashTableT::sentinel_v;
    }

    /// make sure no one else is inserting in the same time!
    template <bool retrieve_index = false>
    constexpr value_t query(const key_t &key, wrapv<retrieve_index> = {}) const noexcept {
      using namespace placeholders;
      int iter = 0;
      int loc = 0;
      size_type bucketOffset = _hf0(key) % _numBuckets * bucket_size;
      for (int iter = 0; iter < 3;) {
        for (loc = 0; loc != bucket_size; ++loc)
          if (_table.keys[bucketOffset + loc].val == key) break;
        if (loc != bucket_size) {
          if constexpr (retrieve_index)
            return bucketOffset + loc;
          else
            return _table.indices[bucketOffset + loc];
        } else {
          ++iter;
          if (iter == 1)
            bucketOffset = _hf1(key) % _numBuckets * bucket_size;
          else if (iter == 2)
            bucketOffset = _hf2(key) % _numBuckets * bucket_size;
        }
      }
      return HashTableT::sentinel_v;
    }
    constexpr size_type entry(const key_t &key) const noexcept {
      using namespace placeholders;
      return (size_type)query(key, true_c);
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
    conditional_t<is_const_structure, const int *, int *> _success{nullptr};
    size_type _numBuckets;
    hasher_type _hf0, _hf1, _hf2;

  protected:
    template <typename VecT,
              enable_if_all<VecT::dim == 1, VecT::extent == dim,
                            std::is_convertible_v<typename VecT::value_type, index_type>> = 0>
    constexpr value_t do_hash(const VecInterface<VecT> &key) const noexcept {
      std::size_t ret = key[0];
      for (int d = 1; d < HashTableT::dim; ++d) hash_combine(ret, key[d]);
      return static_cast<value_t>(ret);
    }
#if defined(__CUDACC__) && ZS_ENABLE_CUDA
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_all<S == execspace_e::cuda, !V> = 0>
    __forceinline__ __host__ __device__ bool atomicSwitchIfEqual(
        status_t *lock, volatile storage_key_type *const dest,
        const storage_key_type &val) noexcept {
      constexpr auto execTag = wrapv<S>{};
      using namespace placeholders;
      const storage_key_type key_sentinel_v = key_t::constant(HashTableT::key_scalar_sentinel_v);
      if constexpr (sizeof(storage_key_type) == 8) {
        static_assert(std::alignment_of_v<storage_key_type> == std::alignment_of_v<u64>,
                      "storage key type alignment is not the same as u64");
        union {
          volatile storage_key_type *const ptr;
          volatile u64 *const ptr64;
        } dst = {dest};
        union {
          const storage_key_type *const ptr;
          const u64 *const ptr64;
        } expected = {&key_sentinel_v};
        union {
          const storage_key_type *const ptr;
          const u64 *const ptr64;
        } desired = {&val};

        return (atomic_cas(exec_cuda, const_cast<u64 *>(dst.ptr64), *expected.ptr64, *desired.ptr64)
                << (storage_key_type::num_padded_bytes * 8))
               == (*expected.ptr64 << (storage_key_type::num_padded_bytes * 8));
      } else if constexpr (sizeof(storage_key_type) == 4) {
        static_assert(std::alignment_of_v<storage_key_type> == std::alignment_of_v<u32>,
                      "storage key type alignment is not the same as u32");
        union {
          volatile storage_key_type *const ptr;
          volatile u32 *const ptr32;
        } dst = {dest};
        union {
          const storage_key_type *const ptr;
          const u32 *const ptr32;
        } expected = {&key_sentinel_v};
        union {
          const storage_key_type *const ptr;
          const u32 *const ptr32;
        } desired = {&val};

        return (atomic_cas(exec_cuda, const_cast<u32 *>(dst.ptr32), *expected.ptr32, *desired.ptr32)
                << (storage_key_type::num_padded_bytes * 8))
               == (*expected.ptr32 << (storage_key_type::num_padded_bytes * 8));
      }
      /// lock
      while (atomic_exch(exec_cuda, lock, 0) == 0)
        ;
      thread_fence(exec_cuda);
      /// cas
      storage_key_type temp;
      for (int d = 0; d != dim; ++d) (void)(temp.val(d) = dest->val.data()[d]);
      bool eqn = temp.val == key_sentinel_v.val;
      if (eqn) {
        for (int d = 0; d != dim; ++d) (void)(dest->val.data()[d] = val.val(d));
      }
      thread_fence(exec_cuda);
      /// unlock
      atomic_exch(exec_cuda, lock, HashTableT::status_sentinel_v);
      return eqn;
    }
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_all<S == execspace_e::cuda, !V> = 0>
    __forceinline__ __host__ __device__ key_t
    atomicLoad(status_t *lock, const volatile storage_key_type *const dest) noexcept {
      constexpr auto execTag = wrapv<S>{};
      using namespace placeholders;
      if constexpr (sizeof(storage_key_type) == 8) {
        static_assert(std::alignment_of_v<storage_key_type> == std::alignment_of_v<u64>,
                      "storage key type alignment is not the same as u64");
        union {
          storage_key_type const volatile *const ptr;
          u64 const volatile *const ptr64;
        } src = {dest};

        storage_key_type result;
        union {
          storage_key_type *const ptr;
          u64 *const ptr64;
        } dst = {&result};

        *dst.ptr64 = *src.ptr64;
        return *dst.ptr;
      } else if constexpr (sizeof(storage_key_type) == 4) {
        static_assert(std::alignment_of_v<storage_key_type> == std::alignment_of_v<u32>,
                      "storage key type alignment is not the same as u32");
        union {
          storage_key_type const volatile *const ptr;
          u32 const volatile *const ptr32;
        } src = {dest};

        storage_key_type result;
        union {
          storage_key_type *const ptr;
          u32 *const ptr32;
        } dst = {&result};

        *dst.ptr32 = *src.ptr32;
        return *dst.ptr;
      }
      /// lock
      while (atomic_exch(exec_cuda, lock, 0) == 0)
        ;
      thread_fence(exec_cuda);
      ///
      key_t return_val;
      for (int d = 0; d != dim; ++d) (void)(return_val.val(d) = dest->val.data()[d]);
      thread_fence(exec_cuda);
      /// unlock
      atomic_exch(exec_cuda, lock, HashTableT::status_sentinel_v);
      return return_val;
    }
#endif
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_all<S != execspace_e::cuda, !V> = 0>
    inline bool atomicSwitchIfEqual(status_t *lock, volatile storage_key_type *const dest,
                                    const storage_key_type &val) noexcept {
      constexpr auto execTag = wrapv<S>{};
      using namespace placeholders;
      const storage_key_type key_sentinel_v = key_t::constant(HashTableT::key_scalar_sentinel_v);
      if constexpr (sizeof(storage_key_type) == 8) {
        static_assert(std::alignment_of_v<storage_key_type> == std::alignment_of_v<u64>,
                      "storage key type alignment is not the same as u64");
        union {
          volatile storage_key_type *const ptr;
          volatile u64 *const ptr64;
        } dst = {dest};
        union {
          const storage_key_type *const ptr;
          const u64 *const ptr64;
        } expected = {&key_sentinel_v};
        union {
          const storage_key_type *const ptr;
          const u64 *const ptr64;
        } desired = {&val};

        return (atomic_cas(execTag, const_cast<u64 *>(dst.ptr64), *expected.ptr64, *desired.ptr64)
                << (storage_key_type::num_padded_bytes * 8))
               == (*expected.ptr64 << (storage_key_type::num_padded_bytes * 8));
      } else if constexpr (sizeof(storage_key_type) == 4) {
        static_assert(std::alignment_of_v<storage_key_type> == std::alignment_of_v<u32>,
                      "storage key type alignment is not the same as u32");
        union {
          volatile storage_key_type *const ptr;
          volatile u32 *const ptr32;
        } dst = {dest};
        union {
          const storage_key_type *const ptr;
          const u32 *const ptr32;
        } expected = {&key_sentinel_v};
        union {
          const storage_key_type *const ptr;
          const u32 *const ptr32;
        } desired = {&val};

        return (atomic_cas(execTag, const_cast<u32 *>(dst.ptr32), *expected.ptr32, *desired.ptr32)
                << (storage_key_type::num_padded_bytes * 8))
               == (*expected.ptr32 << (storage_key_type::num_padded_bytes * 8));
      }
      /// lock
      while (atomic_exch(execTag, lock, 0) == 0)
        ;
      thread_fence(execTag);
      /// cas
      storage_key_type temp;  //= volatile_load(dest);
      for (int d = 0; d != dim; ++d) (void)(temp.val(d) = dest->val.data()[d]);
      bool eqn = temp.val == key_sentinel_v.val;
      if (eqn) {
        for (int d = 0; d != dim; ++d) (void)(dest->val.data()[d] = val.val(d));
      }
      thread_fence(execTag);
      /// unlock
      atomic_exch(execTag, lock, HashTableT::status_sentinel_v);
      return eqn;
    }
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_all<S != execspace_e::cuda, !V> = 0>
    inline key_t atomicLoad(status_t *lock, const volatile storage_key_type *const dest) noexcept {
      constexpr auto execTag = wrapv<S>{};
      using namespace placeholders;
      if constexpr (sizeof(storage_key_type) == 8) {
        static_assert(std::alignment_of_v<storage_key_type> == std::alignment_of_v<u64>,
                      "storage key type alignment is not the same as u64");
        union {
          storage_key_type const volatile *const ptr;
          u64 const volatile *const ptr64;
        } src = {dest};

        storage_key_type result;
        union {
          storage_key_type *const ptr;
          u64 *const ptr64;
        } dst = {&result};

        *dst.ptr64 = *src.ptr64;
        return *dst.ptr;
      } else if constexpr (sizeof(storage_key_type) == 4) {
        static_assert(std::alignment_of_v<storage_key_type> == std::alignment_of_v<u32>,
                      "storage key type alignment is not the same as u32");
        union {
          storage_key_type const volatile *const ptr;
          u32 const volatile *const ptr32;
        } src = {dest};

        storage_key_type result;
        union {
          storage_key_type *const ptr;
          u32 *const ptr32;
        } dst = {&result};

        *dst.ptr32 = *src.ptr32;
        return *dst.ptr;
      }
      /// lock
      while (atomic_exch(execTag, lock, 0) == 0)
        ;
      thread_fence(execTag);
      ///
      key_t return_val;
      for (int d = 0; d != dim; ++d) (void)(return_val.val(d) = dest->val.data()[d]);
      thread_fence(execTag);
      /// unlock
      atomic_exch(execTag, lock, HashTableT::status_sentinel_v);
      return return_val;
    }
  };

  template <execspace_e ExecSpace, typename Tn, int dim, typename Index, int B, typename Allocator>
  constexpr decltype(auto) proxy(bht<Tn, dim, Index, B, Allocator> &table) {
    return BHTView<ExecSpace, bht<Tn, dim, Index, B, Allocator>>{table};
  }
  template <execspace_e ExecSpace, typename Tn, int dim, typename Index, int B, typename Allocator>
  constexpr decltype(auto) proxy(const bht<Tn, dim, Index, B, Allocator> &table) {
    return BHTView<ExecSpace, const bht<Tn, dim, Index, B, Allocator>>{table};
  }

}  // namespace zs
