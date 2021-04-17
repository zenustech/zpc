#pragma once
#include <cuda_runtime_api.h>

#include <type_traits>
#include <zensim/execution/ExecutionPolicy.hpp>

#include "zensim/container/HashTable.hpp"
#include "zensim/math/Hash.hpp"

namespace zs {

  template <typename HashTableT> struct HashTableProxy<execspace_e::cuda, HashTableT> {
    static constexpr int dim = HashTableT::dim;
    using Tn = typename HashTableT::Tn;
    using table_t = typename HashTableT::base_t;
    using key_t = typename HashTableT::key_t;
    using value_t = typename HashTableT::value_t;
    using unsigned_value_t = std::make_unsigned_t<value_t>;
    using status_t = typename HashTableT::status_t;

    constexpr HashTableProxy() = default;
    ~HashTableProxy() = default;

    explicit HashTableProxy(HashTableT &table)
        : _table{table.self()},
          _tableSize{table._tableSize},
          _cnt{table._cnt.data()},
          _activeKeys{table._activeKeys.data()} {}

    __forceinline__ __device__ value_t insert(const key_t &key) {
      using namespace placeholders;
      constexpr key_t key_sentinel_v = key_t::uniform(static_cast<typename key_t::value_type>(-1));
      value_t hashedentry = (do_hash(key) % _tableSize + _tableSize) % _tableSize;
      key_t storedKey = atomicKeyCAS(&_table(_2, hashedentry), &_table(_0, hashedentry), key);
#if 1
      for (; !(storedKey == key_sentinel_v || storedKey == key);) {
        hashedentry = (hashedentry + 127) % _tableSize;
        storedKey = atomicKeyCAS(&_table(_2, hashedentry), &_table(_0, hashedentry), key);
      }
      if (storedKey == key_sentinel_v) {
        auto localno = atomicAdd((unsigned_value_t *)_cnt, (unsigned_value_t)1);
        _table(_1, hashedentry) = localno;
        _activeKeys[localno] = key;
        return localno;  ///< only the one that inserts returns the actual index
      }
#else
      while (!((storedKey = _table(_0, hashedentry)) == key)) {
        if (storedKey == key_sentinel_v) {
          storedKey = atomicKeyCAS(&_table(_2, hashedentry), &_table(_0, hashedentry), key);
        }
        if (_table(_0, hashedentry) == key) {  ///< found entry
          if (storedKey == key_sentinel_v) {   ///< new entry
            auto localno = atomicAdd((unsigned_value_t *)_cnt, (unsigned_value_t)1);
            _table(_1, hashedentry) = localno;
            _activeKeys[localno] = key;
            return localno;
          }
          return _table(_1, hashedentry);
        }
        hashedentry += 127;  ///< search next entry
        if (hashedentry > _tableSize) hashedentry = hashedentry % _tableSize;
      }
#endif
      return HashTableT::sentinel_v;
    }
    /// make sure no one else is inserting in the same time!
    __forceinline__ __device__ value_t query(const key_t &key) const {
      using namespace placeholders;
      value_t hashedentry = (do_hash(key) % _tableSize + _tableSize) % _tableSize;
      while (true) {
        if (key == (key_t)_table(_0, hashedentry)) return _table(_1, hashedentry);
        if (_table(_1, hashedentry) == HashTableT::sentinel_v) return HashTableT::sentinel_v;
        hashedentry += 127;  ///< search next entry
        if (hashedentry > _tableSize) hashedentry = hashedentry % _tableSize;
      }
    }

  protected:
    __forceinline__ __device__ value_t do_hash(const key_t &key) const {
      std::size_t ret = key[0];
      for (int d = 1; d < HashTableT::dim; ++d) hash_combine(ret, key[d]);
      return ret;
    }
    __forceinline__ __device__ key_t atomicKeyCAS(status_t *lock, volatile key_t *const dest,
                                                  const key_t &val) {
      using namespace placeholders;
      constexpr key_t key_sentinel_v = key_t::uniform(static_cast<typename key_t::value_type>(-1));
      key_t return_val;
      int done = 0;
      unsigned int mask = __activemask();
      unsigned int active = __ballot_sync(mask, 1);
      unsigned int done_active = 0;
      while (active != done_active) {
        if (!done) {
          if (atomicCAS(lock, HashTableT::status_sentinel_v, (status_t)0)
              == HashTableT::status_sentinel_v) {
            __threadfence();
            /// <deprecating volatile - JF Bastien - CppCon2019>
            /// access non-volatile using volatile semantics
            /// use cast
            (void)(return_val = *const_cast<key_t *>(dest));
            /// https://github.com/kokkos/kokkos/commit/2fd9fb04a94ecba29a04a0894c99e1d9c16ad66a
            if (return_val == key_sentinel_v) {
              for (int d = 0; d < dim; ++d) (void)(dest->data()[d] = val[d]);
              // (void)(*dest = val);
            }
            __threadfence();
            atomicExch(lock, HashTableT::status_sentinel_v);
            done = 1;
          }
        }
        done_active = __ballot_sync(mask, done);
      }
      return return_val;
    }

    table_t _table;
    const value_t _tableSize;
    value_t *_cnt;
    key_t *_activeKeys;
  };

}  // namespace zs
