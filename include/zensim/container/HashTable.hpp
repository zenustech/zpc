#pragma once
#include "Vector.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/math/Hash.hpp"
#include "zensim/math/Vec.h"
#include "zensim/math/bit/Bits.h"
#include "zensim/memory/MemoryResource.h"
#include "zensim/resource/Resource.h"
#include "zensim/types/Iterator.h"
#include "zensim/types/RuntimeStructurals.hpp"

namespace zs {

  /// Tn:       domain type
  /// Key:      key type
  /// Index:    hash table type (counter type)
  /// Status:   atomic operation state
  template <typename Key, typename Index, typename Status> using hash_table_snode
      = ds::snode_t<ds::decorations<ds::soa>, ds::uniform_domain<0, Index, 1, index_seq<0>>,
                    zs::tuple<wrapt<Key>, wrapt<Index>, Status>, vseq_t<1, 1, 1>>;

  template <typename Key, typename Index, typename Status = int> using hash_table_instance
      = ds::instance_t<ds::dense, hash_table_snode<Key, Index, Status>>;

  template <typename Tn_, int dim_, typename Index> struct HashTable
      : Inherit<Object, HashTable<Tn_, dim_, Index>>,
        hash_table_instance<vec<std::make_signed_t<Tn_>, dim_>, Index, int>,
        MemoryHandle {
    static constexpr int dim = dim_;
    using Tn = std::make_signed_t<Tn_>;
    using key_t = vec<Tn, dim>;
    using value_t = Index;
    using status_t = int;
    using base_t = hash_table_instance<key_t, value_t, status_t>;

    static constexpr Tn key_scalar_sentinel_v = std::numeric_limits<Tn>::max();
    static constexpr value_t sentinel_v{-1};
    static constexpr status_t status_sentinel_v{-1};
    static constexpr std::size_t reserve_ratio_v = 16;

    constexpr MemoryHandle &base() noexcept { return static_cast<MemoryHandle &>(*this); }
    constexpr const MemoryHandle &base() const noexcept {
      return static_cast<const MemoryHandle &>(*this);
    }
    constexpr auto &self() noexcept { return static_cast<base_t &>(*this); }
    constexpr const auto &self() const noexcept { return static_cast<const base_t &>(*this); }

    HashTable(memsrc_e mre = memsrc_e::host, ProcID devid = -1, std::size_t alignment = 0)
        : MemoryHandle{mre, devid},
          base_t{buildInstance(mre, devid, 0)},
          _tableSize{0},
          _cnt{mre, devid, alignment},
          _activeKeys{mre, devid, alignment},
          _align{alignment} {}

    HashTable(std::size_t tableSize, memsrc_e mre = memsrc_e::host, ProcID devid = -1,
              std::size_t alignment = 0)
        : MemoryHandle{mre, devid},
          base_t{buildInstance(mre, devid, next_2pow(tableSize) * reserve_ratio_v)},
          _tableSize{static_cast<value_t>(next_2pow(tableSize) * reserve_ratio_v)},
          _cnt{1, mre, devid, alignment},
          _activeKeys{next_2pow(tableSize) * reserve_ratio_v, mre, devid, alignment},
          _align{alignment} {}
    ~HashTable() {
      if (self().address() && self().node().extent() > 0) self().dealloc();
    }

    HashTable(const HashTable &o)
        : MemoryHandle{o.base()},
          base_t{buildInstance(o.memspace(), o.devid(), o._tableSize)},
          _tableSize{o._tableSize},
          _cnt{o._cnt},
          _activeKeys{o._activeKeys},
          _align{o._align} {
      if (ds::snode_size(o.self().template node<0>()) > 0)
        copy({base(), (void *)self().address()}, {o.base(), (void *)o.self().address()},
             ds::snode_size(o.self().template node<0>()));
    }
    HashTable &operator=(const HashTable &o) {
      if (this == &o) return *this;
      HashTable tmp(o);
      swap(tmp);
      return *this;
    }
    HashTable clone(const MemoryHandle &mh) const {
      HashTable ret{_tableSize / reserve_ratio_v, mh.memspace(), mh.devid(), _align};
      if (_cnt.size() > 0)
        copy({ret._cnt.base(), ret._cnt.data()}, {_cnt.base(), _cnt.data()}, sizeof(value_t));
      if (_activeKeys.size() > 0)
        copy({ret._activeKeys.base(), ret._activeKeys.data()},
             {_activeKeys.base(), _activeKeys.data()}, sizeof(key_t) * _activeKeys.size());
      if (ds::snode_size(self().template node<0>()) > 0)
        copy({ret.base(), (void *)ret.self().address()}, {base(), (void *)self().address()},
             ds::snode_size(self().template node<0>()));
      return ret;
    }

    HashTable(HashTable &&o) noexcept {
      const HashTable defaultTable{};
      base() = std::exchange(o.base(), defaultTable.base());
      self() = std::exchange(o.self(), defaultTable.self());
      _tableSize = std::exchange(o._tableSize, defaultTable._tableSize);
      /// critical! use user-defined move assignment constructor!
      _cnt = std::move(o._cnt);
      _activeKeys = std::move(o._activeKeys);
      _align = std::exchange(o._align, defaultTable._align);
    }
    HashTable &operator=(HashTable &&o) noexcept {
      if (this == &o) return *this;
      HashTable tmp(std::move(o));
      swap(tmp);
      return *this;
    }
    void swap(HashTable &o) noexcept {
      base().swap(o.base());
      std::swap(self(), o.self());
      std::swap(_tableSize, o._tableSize);
      _cnt.swap(o._cnt);
      _activeKeys.swap(o._activeKeys);
      std::swap(_align, o._align);
    }

    inline value_t size() const {
      Vector<value_t> res{1, memsrc_e::host, -1};
      copy({res.base(), res.data()}, {_cnt.base(), _cnt.data()}, sizeof(value_t));
      return res[0];
    }

    value_t _tableSize;
    Vector<value_t> _cnt;
    Vector<key_t> _activeKeys;
    std::size_t _align;

  protected:
    constexpr auto buildInstance(memsrc_e mre, ProcID devid, value_t capacity) const {
      using namespace ds;
      uniform_domain<0, Tn, 1, index_seq<0>> dom{wrapv<0>{}, capacity};
      hash_table_snode<key_t, value_t, status_t> node{
          ds::decorations<ds::soa>{}, dom,
          zs::make_tuple(wrapt<key_t>{}, wrapt<value_t>{}, wrapt<status_t>{}), vseq_t<1, 1, 1>{}};
      auto inst = instance{wrapv<dense>{}, zs::make_tuple(node)};
      if (capacity) inst.alloc(get_memory_source(mre, devid));
      return inst;
    }
  };

#if 0
  using GeneralHashTable = variant<HashTable<i32, 2, int>, HashTable<i32, 2, long long int>,
                                   HashTable<i32, 3, int>, HashTable<i32, 3, long long int>>;
#else
  using GeneralHashTable = variant<HashTable<i32, 3, int>>;
#endif

  template <execspace_e, typename HashTableT, typename = void>
  struct HashTableProxy;  ///< proxy to work within each backends
  template <typename HashTableT> struct HashTableProxy<execspace_e::host, HashTableT> {
    static constexpr int dim = HashTableT::dim;
    using Tn = typename HashTableT::Tn;
    using table_t = typename HashTableT::base_t;
    using key_t = typename HashTableT::key_t;
    using value_t = typename HashTableT::value_t;
    using status_t = typename HashTableT::status_t;

    constexpr HashTableProxy() = default;
    ~HashTableProxy() = default;
    explicit HashTableProxy(HashTableT &table)
        : _table{table.self()},
          _tableSize{table._tableSize},
          _cnt{table._cnt.data()},
          _activeKeys{table._activeKeys.data()} {}

    void clear() {
      using namespace placeholders;
      // reset counter
      *_cnt = 0;
      // reset table
      constexpr key_t key_sentinel_v = key_t::uniform(HashTableT::key_scalar_sentinel_v);
      for (value_t entry = 0; entry < _tableSize; ++entry) {
        _table(_0, entry) = key_sentinel_v;
        _table(_1, entry) = HashTableT::sentinel_v;
      }
    }

    value_t insert(const key_t &key) {
      using namespace placeholders;
      constexpr key_t key_sentinel_v = key_t::uniform(HashTableT::key_scalar_sentinel_v);
      value_t hashedentry = (do_hash(key) % _tableSize + _tableSize) % _tableSize;
      auto storedKey = _table(_0, hashedentry);
      for (; !(storedKey == key_sentinel_v || storedKey == key);) {
        hashedentry = (hashedentry + 127) % _tableSize;
        storedKey = _table(_0, hashedentry);
      }
      auto localno = (*_cnt)++;
      _table(_0, hashedentry) = key;
      _table(_1, hashedentry) = localno;
      _activeKeys[localno] = key;
      if (localno >= _tableSize - 20)
        printf("proximity!!! %lld -> %lld\n", (long long int)localno, (long long int)_tableSize);
      return localno;  ///< only the one that inserts returns the actual index
    }
    value_t query(const key_t &key) const {
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
    constexpr value_t do_hash(const key_t &key) const {
      std::size_t ret = key[0];
      for (int d = 1; d < HashTableT::dim; ++d) hash_combine(ret, key[d]);
      return static_cast<value_t>(ret);
    }
    table_t _table;
    const value_t _tableSize;
    value_t *_cnt;
    key_t *_activeKeys;
  };

  template <execspace_e ExecSpace, typename Tn, int dim, typename Index>
  decltype(auto) proxy(HashTable<Tn, dim, Index> &table) {
    return HashTableProxy<ExecSpace, HashTable<Tn, dim, Index>>{table};
  }

  template <typename ExecPol, typename Tn, int dim, typename Index>
  void refit(ExecPol &&pol, HashTable<Tn, dim, Index> &table);

}  // namespace zs
