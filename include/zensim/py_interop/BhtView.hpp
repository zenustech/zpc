#pragma once
#include "zensim/ZpcMeta.hpp"
#include "zensim/execution/Atomics.hpp"
#include "zensim/math/Vec.h"
#include "zensim/py_interop/HashUtils.hpp"
#include "zensim/py_interop/VectorView.hpp"

namespace zs {

  template <typename Tn_, int dim_, typename Index = int, int B = 32>
  struct BhtViewLite {  // T may be const
    static constexpr bool is_const_structure = is_const<Tn_>::value;
    using index_type = zs::make_signed_t<remove_const_t<Tn_>>;
    using value_type = zs::make_signed_t<remove_const_t<Index>>;
    using status_type = int;
    static constexpr int dim = dim_;
    using key_type = vec<index_type, dim>;
    using size_type = zs::make_unsigned_t<value_type>;

    static constexpr size_type bucket_size = B;
    static constexpr size_type threshold = bucket_size - 2;
    static_assert(threshold > 0, "bucket load threshold must be positive");
    template <typename T> using decorate_t = conditional_t<is_const_structure, add_const_t<T>, T>;

#if 1
    static_assert(sizeof(key_type) == sizeof(index_type) * dim,
                  "storage_key_type assumption of key_type is invalid");
    using storage_key_type = storage_key_type_impl<key_type>;
#else
    struct alignas(next_2pow(sizeof(index_type) * dim)) storage_key_type {
      static constexpr size_t num_total_bytes = next_2pow(sizeof(index_type) * dim);
      static constexpr size_t num_padded_bytes
          = next_2pow(sizeof(index_type) * dim) - sizeof(index_type) * dim;
      constexpr storage_key_type() noexcept = default;
      constexpr storage_key_type(const key_type &k) noexcept : val{k} {}
      constexpr storage_key_type(key_type &&k) noexcept : val{move(k)} {}
      ~storage_key_type() = default;
      constexpr operator key_type &() { return val; }
      constexpr operator const key_type &() const { return val; }
      constexpr operator volatile key_type &() volatile { return val; }
      constexpr operator const volatile key_type &() const volatile { return val; }

      template <typename Key = key_type, typename T = typename Key::value_type,
                enable_if_all<is_vec<Key>::value, Key::extent == 1> = 0>
      constexpr operator T &() {
        return val.val(0);
      }
      template <typename Key = key_type, typename T = typename Key::value_type,
                enable_if_all<is_vec<Key>::value, Key::extent == 1> = 0>
      constexpr operator const T &() const {
        return val.val(0);
      }
      template <typename Key = key_type, typename T = typename Key::value_type,
                enable_if_all<is_vec<Key>::value, Key::extent == 1> = 0>
      constexpr operator volatile T &() volatile {
        return val.data()[0];
      }
      template <typename Key = key_type, typename T = typename Key::value_type,
                enable_if_all<is_vec<Key>::value, Key::extent == 1> = 0>
      constexpr operator const volatile T &() const volatile {
        return val.data()[0];
      }

      key_type val;
    };
#endif

    using hasher_type = universal_hash_base<key_type>;
    // using pair_type = tuple<key_type, value_type>;

    struct table_t {
      VectorViewLite<decorate_t<storage_key_type>> keys{};
      VectorViewLite<decorate_t<value_type>> indices{};
      VectorViewLite<decorate_t<status_type>> status{};
    };

    using unsigned_value_t = make_unsigned_t<value_type>;

    /// @note byte-wise filled with 0x3f
    static constexpr key_type deduce_key_sentinel() noexcept {
      typename key_type::value_type v{0};
      for (int i = 0; i != sizeof(typename key_type::value_type); ++i) v = (v << 8) | 0x3f;
      return key_type::constant(v);
    }
    static constexpr key_type key_sentinel_v = deduce_key_sentinel();
    // static constexpr index_type key_scalar_sentinel_v = detail::deduce_numeric_max<index_type>();
    // // detail::deduce_numeric_max<index_type>();
    static constexpr value_type sentinel_v{-1};  // this requires key_type to be signed type
    static constexpr status_type status_sentinel_v{-1};
    static constexpr value_type failure_token_v = detail::deduce_numeric_lowest<value_type>();

    constexpr BhtViewLite() noexcept = default;
    ~BhtViewLite() = default;
    BhtViewLite(decorate_t<storage_key_type> *const keys, decorate_t<value_type> *const indices,
                decorate_t<status_type> *const status, decorate_t<key_type> *const activeKeys,
                decorate_t<status_type> *const cnt, decorate_t<int> *const success,
                size_type tableSize, const hasher_type &hf0, const hasher_type &hf1,
                const hasher_type &hf2) noexcept
        : _table{keys, indices, status},
          _activeKeys{activeKeys},
          _cnt{cnt},
          _success{success},
          _tableSize{tableSize},
          _numBuckets{tableSize / bucket_size},
          _hf0{hf0},
          _hf1{hf1},
          _hf2{hf2} {}

    constexpr auto size() const noexcept { return *_cnt; }
    ///
    /// @note cuda
    ///
#if ZS_ENABLE_CUDA || ZS_ENABLE_MUSA || ZS_ENABLE_ROCM
#  if defined(__CUDACC__) || defined(__MUSACC__) || defined(__HIPCC__)
    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    __forceinline__ __device__ value_type insert(const key_type &key,
                                                 value_type insertion_index = sentinel_v,
                                                 bool enqueueKey = true) noexcept {
      if (_numBuckets == 0) return failure_token_v;
      constexpr auto key_sentinel_v = hash_table_type::deduce_key_sentinel();
      int iter = 0;
      int load = 0;
      size_type bucketOffset = _hf0(key) % _numBuckets * bucket_size;
      while (iter < 3) {
        for (; load != bucket_size; ++load) {
          key_type curKey = atomicLoad(
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
          key_type storedKey = key_sentinel_v;
          if (atomicSwitchIfEqual(
                  &_table.status[bucketOffset + load],
                  const_cast<volatile storage_key_type *>(&_table.keys[bucketOffset + load]),
                  key)) {
            auto localno = insertion_index;
            if (insertion_index == sentinel_v)
              localno
                  = (value_type)atomic_add(exectag, (unsigned_value_t *)_cnt, (unsigned_value_t)1);
            _table.indices[bucketOffset + load] = localno;
            if (enqueueKey) _activeKeys[localno] = key;
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
      return failure_token_v;
    }
    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    __forceinline__ __device__ value_type tile_insert(
        cooperative_groups::thread_block_tile<bucket_size, cooperative_groups::thread_block> &tile,
        const key_type &key, index_type insertion_index = sentinel_v,
        bool enqueueKey = true) noexcept {
      if (_numBuckets == 0) return failure_token_v;
      constexpr auto key_sentinel_v = hash_table_type::deduce_key_sentinel();
      int iter = 0;
      size_type bucketOffset = _hf0(key) % _numBuckets * bucket_size;
      while (iter < 3) {
        key_type laneKey = atomicTileLoad(tile, &_table.status[bucketOffset],
                                          const_cast<const volatile storage_key_type *>(
                                              &_table.keys[bucketOffset + tile.thread_rank()]));
        if (tile.any(laneKey == key)) return HashTableT::sentinel_v;
        auto loadMap = tile.ballot(laneKey != key_sentinel_v);
        auto load = __popc(loadMap);
        if (load <= threshold) {
          int switched = 0;
          value_type localno = insertion_index;
          if (tile.thread_rank() == 0) {
            // by the time, position [load] may already be filled
            switched = atomicSwitchIfEqual(
                &_table.status[bucketOffset],
                const_cast<volatile storage_key_type *>(&_table.keys[bucketOffset + load]), key);
            if (switched) {
              if (insertion_index == sentinel_v)
                localno = (value_type)atomic_add(exectag, (unsigned_value_t *)_cnt,
                                                 (unsigned_value_t)1);
              _table.indices[bucketOffset + load] = localno;
              if (enqueueKey) _activeKeys[localno] = key;
              if (localno >= _tableSize - 20) {
                printf("proximity!!! %d -> %d\n", (int)localno, (int)_tableSize);
                *_success = false;
                localno = failure_token_v;
              }
            }
          }
          switched = tile.shfl(switched, 0);
          if (switched) {
            localno = tile.shfl(localno, 0);
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
      return failure_token_v;
    }
    template <bool retrieve_index = true> __forceinline__ __device__ value_type tile_query(
        cooperative_groups::thread_block_tile<bucket_size, cooperative_groups::thread_block> &tile,
        const key_type &key, wrapv<retrieve_index> = {}) const noexcept {
      using namespace placeholders;
      if (_numBuckets == 0) {
        if constexpr (retrieve_index)
          return HashTableT::sentinel_v;
        else
          return detail::deduce_numeric_max<value_type>();
      }
      int loc = 0;
      size_type bucketOffset = _hf0(key) % _numBuckets * bucket_size;
      for (int iter = 0; iter < 3;) {
        key_type laneKey = _table.keys[bucketOffset + tile.thread_rank()];
        auto keyExistMap = tile.ballot(laneKey == key);
        loc = __ffs(keyExistMap) - 1;
        if (loc != -1) {
          if constexpr (retrieve_index)
            return _table.indices[bucketOffset + loc];
          else
            return bucketOffset + loc;
        } else {
          ++iter;
          if (iter == 1)
            bucketOffset = _hf1(key) % _numBuckets * bucket_size;
          else if (iter == 2)
            bucketOffset = _hf2(key) % _numBuckets * bucket_size;
        }
      }
      if constexpr (retrieve_index)
        return HashTableT::sentinel_v;
      else
        return detail::deduce_numeric_max<value_type>();
    }

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    __forceinline__ __device__ bool atomicSwitchIfEqual(status_type *lock,
                                                        volatile storage_key_type *const dest,
                                                        const storage_key_type &val) noexcept {
      using namespace placeholders;
      constexpr auto space = deduce_execution_space();
      constexpr auto key_sentinel_v = hash_table_type::deduce_key_sentinel();
      const storage_key_type storage_key_sentinel_v = key_sentinel_v;
      if constexpr (sizeof(storage_key_type) == 8) {
        static_assert(alignof(storage_key_type) == alignof(u64),
                      "storage key type alignment is not the same as u64");
        union {
          volatile storage_key_type *const ptr;
          volatile u64 *const ptr64;
        } dst = {dest};
        union {
          const storage_key_type *const ptr;
          const u64 *const ptr64;
        } expected = {&storage_key_sentinel_v};
        union {
          const storage_key_type *const ptr;
          const u64 *const ptr64;
        } desired = {&val};

        return (atomic_cas(wrapv<space>{}, const_cast<u64 *>(dst.ptr64), *expected.ptr64,
                           *desired.ptr64)
                << (storage_key_type::num_padded_bytes * 8))
               == (*expected.ptr64 << (storage_key_type::num_padded_bytes * 8));
      } else if constexpr (sizeof(storage_key_type) == 4) {
        static_assert(alignof(storage_key_type) == alignof(u32),
                      "storage key type alignment is not the same as u32");
        union {
          volatile storage_key_type *const ptr;
          volatile u32 *const ptr32;
        } dst = {dest};
        union {
          const storage_key_type *const ptr;
          const u32 *const ptr32;
        } expected = {&storage_key_sentinel_v};
        union {
          const storage_key_type *const ptr;
          const u32 *const ptr32;
        } desired = {&val};

        return (atomic_cas(wrapv<space>{}, const_cast<u32 *>(dst.ptr32), *expected.ptr32,
                           *desired.ptr32)
                << (storage_key_type::num_padded_bytes * 8))
               == (*expected.ptr32 << (storage_key_type::num_padded_bytes * 8));
      }
      /// lock
      while (atomic_exch(wrapv<space>{}, lock, 0) == 0);
      thread_fence(wrapv<space>{});
      /// cas
      storage_key_type temp;
      for (int d = 0; d != dim; ++d) (void)(temp.val(d) = dest->val.data()[d]);
      bool eqn = temp.val == key_sentinel_v;
      if (eqn) {
        for (int d = 0; d != dim; ++d) (void)(dest->val.data()[d] = val.val(d));
      }
      thread_fence(wrapv<space>{});
      /// unlock
      atomic_exch(wrapv<space>{}, lock, HashTableT::status_sentinel_v);
      return eqn;
    }
    /// @ref https://stackoverflow.com/questions/32341081/how-to-have-atomic-load-in-cuda
    template <bool V = is_const_structure, enable_if_t<!V> = 0> __forceinline__ __device__ key_type
    atomicLoad(status_type *lock, const volatile storage_key_type *const dest) noexcept {
      using namespace placeholders;
      constexpr auto space = deduce_execution_space();
      if constexpr (sizeof(storage_key_type) == 8) {
        thread_fence(wrapv<space>{});
        static_assert(alignof(storage_key_type) == alignof(u64),
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

        /// @note beware of the potential torn read issue
        // *dst.ptr64 = atomic_or(wrapv<space>{}, const_cast<u64 *>(src.ptr64), (u64)0);
        *dst.ptr64 = *src.ptr64;
        thread_fence(wrapv<space>{});
        return *dst.ptr;
      } else if constexpr (sizeof(storage_key_type) == 4) {
        thread_fence(wrapv<space>{});
        static_assert(alignof(storage_key_type) == alignof(u32),
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

        /// @note beware of the potential torn read issue
        // *dst.ptr32 = atomic_or(wrapv<space>{}, const_cast<u32 *>(src.ptr32), (u32)0);
        *dst.ptr32 = *src.ptr32;
        thread_fence(wrapv<space>{});
        return *dst.ptr;
      }
      /// lock
      while (atomic_exch(wrapv<space>{}, lock, 0) == 0);
      thread_fence(wrapv<space>{});
      ///
      key_type return_val;
      for (int d = 0; d != dim; ++d) (void)(return_val.val(d) = dest->val.data()[d]);
      thread_fence(wrapv<space>{});
      /// unlock
      atomic_exch(wrapv<space>{}, lock, HashTableT::status_sentinel_v);
      return return_val;
    }
    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    __forceinline__ __device__ key_type atomicTileLoad(
        cooperative_groups::thread_block_tile<bucket_size, cooperative_groups::thread_block> &tile,
        status_type *lock, const volatile storage_key_type *const dest) noexcept {
      using namespace placeholders;
      constexpr auto space = deduce_execution_space();
      if constexpr (sizeof(storage_key_type) == 8) {
        thread_fence(wrapv<space>{});
        static_assert(alignof(storage_key_type) == alignof(u64),
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

        /// @note beware of the potential torn read issue
        // *dst.ptr64 = atomic_or(wrapv<space>{}, const_cast<u64 *>(src.ptr64), (u64)0);
        *dst.ptr64 = *src.ptr64;
        thread_fence(wrapv<space>{});
        return *dst.ptr;
      } else if constexpr (sizeof(storage_key_type) == 4) {
        thread_fence(wrapv<space>{});
        static_assert(alignof(storage_key_type) == alignof(u32),
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

        /// @note beware of the potential torn read issue
        // *dst.ptr32 = atomic_or(wrapv<space>{}, const_cast<u32 *>(src.ptr32), (u32)0);
        *dst.ptr32 = *src.ptr32;
        thread_fence(wrapv<space>{});
        return *dst.ptr;
      }
      /// lock
      if (tile.thread_rank() == 0)
        while (atomic_exch(wrapv<space>{}, lock, 0) == 0);
      tile.sync();
      thread_fence(wrapv<space>{});
      ///
      key_type return_val;
      for (int d = 0; d != dim; ++d) (void)(return_val.val(d) = dest->val.data()[d]);
      thread_fence(wrapv<space>{});
      /// unlock
      if (tile.thread_rank() == 0) atomic_exch(wrapv<space>{}, lock, HashTableT::status_sentinel_v);
      tile.sync();
      return return_val;
    }
#  endif
    ///
    /// @note host
    ///
#elif ZS_ENABLE_OPENMP
    ///
    /// @note openmp
    ///
    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    inline value_type insert(const key_type &key, value_type insertion_index = sentinel_v,
                             bool enqueueKey = true) noexcept {
      if (_numBuckets == 0) return failure_token_v;
      constexpr auto key_sentinel_v = deduce_key_sentinel();
      int iter = 0;
      int load = 0;
      size_type bucketOffset = _hf0(key) % _numBuckets * bucket_size;
      while (iter < 3) {
        for (; load != bucket_size; ++load) {
          key_type curKey = atomicLoad(
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
          key_type storedKey = key_sentinel_v;
          if (atomicSwitchIfEqual(
                  &_table.status[bucketOffset + load],
                  const_cast<volatile storage_key_type *>(&_table.keys[bucketOffset + load]),
                  key)) {
            auto localno = insertion_index;
            if (insertion_index == sentinel_v)
              localno
                  = (value_type)atomic_add(omp_c, (unsigned_value_t *)_cnt, (unsigned_value_t)1);
            _table.indices[bucketOffset + load] = localno;
            if (enqueueKey) _activeKeys[localno] = key;
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
      return failure_token_v;
    }

    /// make sure no one else is inserting in the same time!
    template <bool retrieve_index = true>
    constexpr value_type query(const key_type &key, wrapv<retrieve_index> = {}) const noexcept {
      using namespace placeholders;
      if (_numBuckets == 0) {
        if constexpr (retrieve_index)
          return sentinel_v;
        else
          return detail::deduce_numeric_max<value_type>();
      }
      int loc = 0;
      size_type bucketOffset = _hf0(key) % _numBuckets * bucket_size;
      for (int iter = 0; iter < 3;) {
        for (loc = 0; loc != bucket_size; ++loc)
          if (_table.keys[bucketOffset + loc].val == key) break;
        if (loc != bucket_size) {
          if constexpr (retrieve_index)
            return _table.indices[bucketOffset + loc];
          else
            return bucketOffset + loc;
        } else {
          ++iter;
          if (iter == 1)
            bucketOffset = _hf1(key) % _numBuckets * bucket_size;
          else if (iter == 2)
            bucketOffset = _hf2(key) % _numBuckets * bucket_size;
        }
      }
      if constexpr (retrieve_index)
        return sentinel_v;
      else
        return detail::deduce_numeric_max<value_type>();
    }
    constexpr size_type entry(const key_type &key) const noexcept {
      using namespace placeholders;
      return static_cast<size_type>(query(key, true_c));
    }

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    inline bool atomicSwitchIfEqual(status_type *lock, volatile storage_key_type *const dest,
                                    const storage_key_type &val) noexcept {
      using namespace placeholders;
      constexpr auto key_sentinel_v = deduce_key_sentinel();
      const storage_key_type storage_key_sentinel_v = key_sentinel_v;
      if constexpr (sizeof(storage_key_type) == 8) {
        static_assert(alignof(storage_key_type) == alignof(u64),
                      "storage key type alignment is not the same as u64");
        union {
          volatile storage_key_type *const ptr;
          volatile u64 *const ptr64;
        } dst = {dest};
        union {
          const storage_key_type *const ptr;
          const u64 *const ptr64;
        } expected = {&storage_key_sentinel_v};
        union {
          const storage_key_type *const ptr;
          const u64 *const ptr64;
        } desired = {&val};

        return (atomic_cas(omp_c, const_cast<u64 *>(dst.ptr64), *expected.ptr64, *desired.ptr64)
                << (storage_key_type::num_padded_bytes * 8))
               == (*expected.ptr64 << (storage_key_type::num_padded_bytes * 8));
      } else if constexpr (sizeof(storage_key_type) == 4) {
        static_assert(alignof(storage_key_type) == alignof(u32),
                      "storage key type alignment is not the same as u32");
        union {
          volatile storage_key_type *const ptr;
          volatile u32 *const ptr32;
        } dst = {dest};
        union {
          const storage_key_type *const ptr;
          const u32 *const ptr32;
        } expected = {&storage_key_sentinel_v};
        union {
          const storage_key_type *const ptr;
          const u32 *const ptr32;
        } desired = {&val};

        return (atomic_cas(omp_c, const_cast<u32 *>(dst.ptr32), *expected.ptr32, *desired.ptr32)
                << (storage_key_type::num_padded_bytes * 8))
               == (*expected.ptr32 << (storage_key_type::num_padded_bytes * 8));
      }
      /// lock
      while (atomic_exch(omp_c, lock, 0) == 0);
      thread_fence(omp_c);
      /// cas
      storage_key_type temp;  //= volatile_load(dest);
      for (int d = 0; d != dim; ++d) (void)(temp.val(d) = dest->val.data()[d]);
      bool eqn = temp.val == key_sentinel_v;
      if (eqn) {
        for (int d = 0; d != dim; ++d) (void)(dest->val.data()[d] = val.val(d));
      }
      thread_fence(omp_c);
      /// unlock
      atomic_exch(omp_c, lock, status_sentinel_v);
      return eqn;
    }
    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    inline key_type atomicLoad(status_type *lock,
                               const volatile storage_key_type *const dest) noexcept {
      using namespace placeholders;
      if constexpr (sizeof(storage_key_type) == 8) {
        static_assert(alignof(storage_key_type) == alignof(u64),
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
        static_assert(alignof(storage_key_type) == alignof(u32),
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
      while (atomic_exch(omp_c, lock, 0) == 0);
      thread_fence(omp_c);
      ///
      key_type return_val;
      for (int d = 0; d != dim; ++d) (void)(return_val.val(d) = dest->val.data()[d]);
      thread_fence(omp_c);
      /// unlock
      atomic_exch(omp_c, lock, status_sentinel_v);
      return return_val;
    }
#else
    ///
    /// @note sequential
    ///
    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    inline value_type insert(const key_type &key, value_type insertion_index = sentinel_v,
                             bool enqueueKey = true) noexcept {
      if (_numBuckets == 0) return failure_token_v;
      constexpr auto key_sentinel_v = deduce_key_sentinel();
      int iter = 0;
      int load = 0;
      size_type bucketOffset = _hf0(key) % _numBuckets * bucket_size;
      while (iter < 3) {
        for (; load != bucket_size; ++load) {
          key_type curKey = atomicLoad(
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
          key_type storedKey = key_sentinel_v;
          if (atomicSwitchIfEqual(
                  &_table.status[bucketOffset + load],
                  const_cast<volatile storage_key_type *>(&_table.keys[bucketOffset + load]),
                  key)) {
            auto localno = insertion_index;
            if (insertion_index == sentinel_v)
              localno
                  = (value_type)atomic_add(seq_c, (unsigned_value_t *)_cnt, (unsigned_value_t)1);
            _table.indices[bucketOffset + load] = localno;
            if (enqueueKey) _activeKeys[localno] = key;
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
      return failure_token_v;
    }

    /// make sure no one else is inserting in the same time!
    template <bool retrieve_index = true>
    constexpr value_type query(const key_type &key, wrapv<retrieve_index> = {}) const noexcept {
      using namespace placeholders;
      if (_numBuckets == 0) {
        if constexpr (retrieve_index)
          return sentinel_v;
        else
          return detail::deduce_numeric_max<value_type>();
      }
      int loc = 0;
      size_type bucketOffset = _hf0(key) % _numBuckets * bucket_size;
      for (int iter = 0; iter < 3;) {
        for (loc = 0; loc != bucket_size; ++loc)
          if (_table.keys[bucketOffset + loc].val == key) break;
        if (loc != bucket_size) {
          if constexpr (retrieve_index)
            return _table.indices[bucketOffset + loc];
          else
            return bucketOffset + loc;
        } else {
          ++iter;
          if (iter == 1)
            bucketOffset = _hf1(key) % _numBuckets * bucket_size;
          else if (iter == 2)
            bucketOffset = _hf2(key) % _numBuckets * bucket_size;
        }
      }
      if constexpr (retrieve_index)
        return sentinel_v;
      else
        return detail::deduce_numeric_max<value_type>();
    }
    constexpr size_type entry(const key_type &key) const noexcept {
      using namespace placeholders;
      return static_cast<size_type>(query(key, true_c));
    }

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    inline bool atomicSwitchIfEqual(status_type *lock, volatile storage_key_type *const dest,
                                    const storage_key_type &val) noexcept {
      using namespace placeholders;
      constexpr auto key_sentinel_v = deduce_key_sentinel();
      const storage_key_type storage_key_sentinel_v = key_sentinel_v;
      if constexpr (sizeof(storage_key_type) == 8) {
        static_assert(alignof(storage_key_type) == alignof(u64),
                      "storage key type alignment is not the same as u64");
        union {
          volatile storage_key_type *const ptr;
          volatile u64 *const ptr64;
        } dst = {dest};
        union {
          const storage_key_type *const ptr;
          const u64 *const ptr64;
        } expected = {&storage_key_sentinel_v};
        union {
          const storage_key_type *const ptr;
          const u64 *const ptr64;
        } desired = {&val};

        return (atomic_cas(seq_c, const_cast<u64 *>(dst.ptr64), *expected.ptr64, *desired.ptr64)
                << (storage_key_type::num_padded_bytes * 8))
               == (*expected.ptr64 << (storage_key_type::num_padded_bytes * 8));
      } else if constexpr (sizeof(storage_key_type) == 4) {
        static_assert(alignof(storage_key_type) == alignof(u32),
                      "storage key type alignment is not the same as u32");
        union {
          volatile storage_key_type *const ptr;
          volatile u32 *const ptr32;
        } dst = {dest};
        union {
          const storage_key_type *const ptr;
          const u32 *const ptr32;
        } expected = {&storage_key_sentinel_v};
        union {
          const storage_key_type *const ptr;
          const u32 *const ptr32;
        } desired = {&val};

        return (atomic_cas(seq_c, const_cast<u32 *>(dst.ptr32), *expected.ptr32, *desired.ptr32)
                << (storage_key_type::num_padded_bytes * 8))
               == (*expected.ptr32 << (storage_key_type::num_padded_bytes * 8));
      }
      /// lock
      while (atomic_exch(seq_c, lock, 0) == 0);
      thread_fence(seq_c);
      /// cas
      storage_key_type temp;  //= volatile_load(dest);
      for (int d = 0; d != dim; ++d) (void)(temp.val(d) = dest->val.data()[d]);
      bool eqn = temp.val == key_sentinel_v;
      if (eqn) {
        for (int d = 0; d != dim; ++d) (void)(dest->val.data()[d] = val.val(d));
      }
      thread_fence(seq_c);
      /// unlock
      atomic_exch(seq_c, lock, status_sentinel_v);
      return eqn;
    }
    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    inline key_type atomicLoad(status_type *lock,
                               const volatile storage_key_type *const dest) noexcept {
      using namespace placeholders;
      if constexpr (sizeof(storage_key_type) == 8) {
        static_assert(alignof(storage_key_type) == alignof(u64),
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
        static_assert(alignof(storage_key_type) == alignof(u32),
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
      while (atomic_exch(seq_c, lock, 0) == 0);
      thread_fence(seq_c);
      ///
      key_type return_val;
      for (int d = 0; d != dim; ++d) (void)(return_val.val(d) = dest->val.data()[d]);
      thread_fence(seq_c);
      /// unlock
      atomic_exch(seq_c, lock, status_sentinel_v);
      return return_val;
    }
#endif

    ///
    /// @note sequential
    ///
#ifdef ZS_ENABLE_SEQUENTIAL
    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    inline value_type insert(const key_type &key, value_type insertion_index = sentinel_v,
                             bool enqueueKey = true) noexcept {
      if (_numBuckets == 0) return failure_token_v;
      constexpr auto key_sentinel_v = deduce_key_sentinel();
      int iter = 0;
      int load = 0;
      size_type bucketOffset = _hf0(key) % _numBuckets * bucket_size;
      while (iter < 3) {
        for (; load != bucket_size; ++load) {
          key_type curKey = atomicLoad(
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
          key_type storedKey = key_sentinel_v;
          if (atomicSwitchIfEqual(
                  &_table.status[bucketOffset + load],
                  const_cast<volatile storage_key_type *>(&_table.keys[bucketOffset + load]),
                  key)) {
            auto localno = insertion_index;
            if (insertion_index == sentinel_v)
              localno
                  = (value_type)atomic_add(seq_c, (unsigned_value_t *)_cnt, (unsigned_value_t)1);
            _table.indices[bucketOffset + load] = localno;
            if (enqueueKey) _activeKeys[localno] = key;
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
      return failure_token_v;
    }

    /// make sure no one else is inserting in the same time!
    template <bool retrieve_index = true>
    constexpr value_type query(const key_type &key, wrapv<retrieve_index> = {}) const noexcept {
      using namespace placeholders;
      if (_numBuckets == 0) {
        if constexpr (retrieve_index)
          return sentinel_v;
        else
          return detail::deduce_numeric_max<value_type>();
      }
      int loc = 0;
      size_type bucketOffset = _hf0(key) % _numBuckets * bucket_size;
      for (int iter = 0; iter < 3;) {
        for (loc = 0; loc != bucket_size; ++loc)
          if (_table.keys[bucketOffset + loc].val == key) break;
        if (loc != bucket_size) {
          if constexpr (retrieve_index)
            return _table.indices[bucketOffset + loc];
          else
            return bucketOffset + loc;
        } else {
          ++iter;
          if (iter == 1)
            bucketOffset = _hf1(key) % _numBuckets * bucket_size;
          else if (iter == 2)
            bucketOffset = _hf2(key) % _numBuckets * bucket_size;
        }
      }
      if constexpr (retrieve_index)
        return sentinel_v;
      else
        return detail::deduce_numeric_max<value_type>();
    }
    constexpr size_type entry(const key_type &key) const noexcept {
      using namespace placeholders;
      return static_cast<size_type>(query(key, true_c));
    }

    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    inline bool atomicSwitchIfEqual(status_type *lock, volatile storage_key_type *const dest,
                                    const storage_key_type &val) noexcept {
      using namespace placeholders;
      constexpr auto key_sentinel_v = deduce_key_sentinel();
      const storage_key_type storage_key_sentinel_v = key_sentinel_v;
      if constexpr (sizeof(storage_key_type) == 8) {
        static_assert(alignof(storage_key_type) == alignof(u64),
                      "storage key type alignment is not the same as u64");
        union {
          volatile storage_key_type *const ptr;
          volatile u64 *const ptr64;
        } dst = {dest};
        union {
          const storage_key_type *const ptr;
          const u64 *const ptr64;
        } expected = {&storage_key_sentinel_v};
        union {
          const storage_key_type *const ptr;
          const u64 *const ptr64;
        } desired = {&val};

        return (atomic_cas(seq_c, const_cast<u64 *>(dst.ptr64), *expected.ptr64, *desired.ptr64)
                << (storage_key_type::num_padded_bytes * 8))
               == (*expected.ptr64 << (storage_key_type::num_padded_bytes * 8));
      } else if constexpr (sizeof(storage_key_type) == 4) {
        static_assert(alignof(storage_key_type) == alignof(u32),
                      "storage key type alignment is not the same as u32");
        union {
          volatile storage_key_type *const ptr;
          volatile u32 *const ptr32;
        } dst = {dest};
        union {
          const storage_key_type *const ptr;
          const u32 *const ptr32;
        } expected = {&storage_key_sentinel_v};
        union {
          const storage_key_type *const ptr;
          const u32 *const ptr32;
        } desired = {&val};

        return (atomic_cas(seq_c, const_cast<u32 *>(dst.ptr32), *expected.ptr32, *desired.ptr32)
                << (storage_key_type::num_padded_bytes * 8))
               == (*expected.ptr32 << (storage_key_type::num_padded_bytes * 8));
      }
      /// lock
      while (atomic_exch(seq_c, lock, 0) == 0);
      thread_fence(seq_c);
      /// cas
      storage_key_type temp;  //= volatile_load(dest);
      for (int d = 0; d != dim; ++d) (void)(temp.val(d) = dest->val.data()[d]);
      bool eqn = temp.val == key_sentinel_v;
      if (eqn) {
        for (int d = 0; d != dim; ++d) (void)(dest->val.data()[d] = val.val(d));
      }
      thread_fence(seq_c);
      /// unlock
      atomic_exch(seq_c, lock, status_sentinel_v);
      return eqn;
    }
    template <bool V = is_const_structure, enable_if_t<!V> = 0>
    inline key_type atomicLoad(status_type *lock,
                               const volatile storage_key_type *const dest) noexcept {
      using namespace placeholders;
      if constexpr (sizeof(storage_key_type) == 8) {
        static_assert(alignof(storage_key_type) == alignof(u64),
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
        static_assert(alignof(storage_key_type) == alignof(u32),
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
      while (atomic_exch(seq_c, lock, 0) == 0);
      thread_fence(seq_c);
      ///
      key_type return_val;
      for (int d = 0; d != dim; ++d) (void)(return_val.val(d) = dest->val.data()[d]);
      thread_fence(seq_c);
      /// unlock
      atomic_exch(seq_c, lock, status_sentinel_v);
      return return_val;
    }
#endif

    table_t _table{};
    VectorViewLite<decorate_t<key_type>> _activeKeys{};
    // conditional_t<is_const_structure, const key_type *, key_type *> _activeKeys{nullptr};
    VectorViewLite<decorate_t<value_type>> _cnt{};
    // conditional_t<is_const_structure, const value_type *, value_type *> _cnt{nullptr};
    VectorViewLite<decorate_t<int>> _success{};
    // conditional_t<is_const_structure, const int *, int *> _success{nullptr};
    size_type _tableSize{0}, _numBuckets{};  // constness makes non-trivial
    hasher_type _hf0, _hf1, _hf2;
  };

}  // namespace zs