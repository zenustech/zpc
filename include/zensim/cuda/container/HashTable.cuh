#pragma once
#include <zensim/cuda/Cuda.h>
#include <zensim/math/Vec.h>
#include <zensim/types/Object.h>

#include <zensim/types/RuntimeStructurals.hpp>

namespace zs {

  /// chenyao lou
  /// https://preshing.com/20130605/the-worlds-simplest-lock-free-hash-table/
  ///

  template <typename Tn, typename Key, typename Index, typename Status> using hash_table_snode
      = ds::snode_t<ds::decorations<ds::soa>, ds::uniform_domain<0, Tn, 1, index_seq<0>>,
                    zs::tuple<wrapt<Key>, wrapt<Index>, Status>, vseq_t<1, 1, 1>>;

  template <typename Tn, typename Key, typename Index, typename Status = int>
  using hash_table_instance = ds::instance_t<ds::dense, hash_table_snode<Tn, Key, Index, Status>>;

  template <typename Tn_, int dim, typename Index, typename Fn> struct HashTable
      : Inherit<Object, HashTable<Tn_, dim, Index, Fn>>,
        hash_table_instance<Index, vec<Tn_, dim>, Index, int> {
    using Tn = Tn_;
    using key_t = vec<Tn, dim>;
    using value_t = Index;
    using status_t = int;
    using base_t = hash_table_instance<value_t, key_t, value_t, status_t>;

    static constexpr value_t sentinel_v{-1};
    static constexpr status_t status_sentinel_v{-1};

  protected:
    constexpr auto &self() noexcept { return static_cast<base_t &>(*this); }
    constexpr const auto &self() const noexcept { return static_cast<const base_t &>(*this); }

    template <typename Allocator>
    constexpr auto buildInstance(Allocator allocator, value_t entryCount) {
      using namespace ds;
      uniform_domain<0, Tn, 1, index_seq<0>> dom{wrapv<0>{}, entryCount};
      hash_table_snode<value_t, key_t, value_t, status_t> node{
          ds::decorations<ds::soa>{}, dom,
          zs::make_tuple(wrapt<key_t>{}, wrapt<value_t>{}, wrapt<status_t>{}), vseq_t<1, 1, 1>{}};
      auto inst = instance{wrapv<dense>{}, zs::make_tuple(node)};
      inst.alloc(allocator, Cuda::alignment());
      return inst;
    }

  public:
    template <typename Allocator> HashTable(Allocator allocator, const Fn &hf, value_t tableSize)
        : base_t{buildInstance(allocator, tableSize)},
          _hf{hf},
          _tableSize{tableSize},
          _cnt{(value_t *)allocator.allocate(sizeof(value_t))},
          _activeKeys{(key_t *)allocator.allocate(sizeof(key_t) * tableSize)} {}
    ~HashTable() {}

    void reset(void *stream) {
      resetTable(stream);
      Cuda::driver().memsetAsync(_cnt, 0, sizeof(value_t), stream);
    }
    void resetTable(void *stream) {
      Cuda::driver().memsetAsync(this->getHandles().template get<0>(), 0xff, this->node().size(),
                                 stream);
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
          if (atomicCAS(lock, status_sentinel_v, (status_t)0) == status_sentinel_v) {
            __threadfence();
            /// <deprecating volatile - JF Bastien - CppCon2019>
            /// access non-volatile using volatile semantics
            /// use cast
            return_val = *const_cast<key_t *>(dest);
            /// https://github.com/kokkos/kokkos/commit/2fd9fb04a94ecba29a04a0894c99e1d9c16ad66a
            if (return_val == key_sentinel_v) (void)(*dest = val);
            __threadfence();
            atomicExch(lock, status_sentinel_v);
            done = 1;
          }
        }
        done_active = __ballot_sync(mask, done);
      }
      return return_val;
    }
    __forceinline__ __device__ value_t insert(const key_t &key) {
      using namespace placeholders;
      constexpr key_t key_sentinel_v = key_t::uniform(static_cast<typename key_t::value_type>(-1));
      value_t hashedentry = (_hf(key) % _tableSize + _tableSize) % _tableSize;
      key_t storedKey = atomicKeyCAS(&(*this)(_2, hashedentry), &(*this)(_0, hashedentry), key);
      for (; !(storedKey == key_sentinel_v || storedKey == key);) {
        hashedentry = (hashedentry + 127) % _tableSize;
        storedKey = atomicKeyCAS(&(*this)(_2, hashedentry), &(*this)(_0, hashedentry), key);
      }
      if (storedKey == key_sentinel_v) {
        auto localno = atomicAdd(_cnt, 1);
        (*this)(_1, hashedentry) = localno;
        _activeKeys[localno] = key;
        return localno;  ///< only the one that inserts returns the actual index
      }
      return sentinel_v;
    }
    /// make sure no one else is inserting in the same time!
    __forceinline__ __device__ value_t query(const key_t &key) const {
      using namespace placeholders;
      value_t hashedentry = (_hf(key) % _tableSize + _tableSize) % _tableSize;
      while (true) {
        if (key == (key_t)(*this)(_0, hashedentry)) return (*this)(_1, hashedentry);
        if ((*this)(_1, hashedentry) == sentinel_v) return sentinel_v;
        hashedentry += 127;  ///< search next entry
        if (hashedentry > _tableSize) hashedentry = hashedentry % _tableSize;
      }
    }

    Fn _hf;
    value_t _tableSize;
    value_t *_cnt;
    key_t *_activeKeys;
  };

}  // namespace zs
