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

  template <typename Tn_, int dim_, typename Index> struct HashTable : MemoryHandle {
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
    constexpr auto &self() noexcept { return _table; }
    constexpr const auto &self() const noexcept { return _table; }

    HashTable(memsrc_e mre = memsrc_e::host, ProcID devid = -1, std::size_t alignment = 0)
        : _table{buildInstance(mre, devid, 0)},
          MemoryHandle{mre, devid},
          _tableSize{0},
          _cnt{mre, devid, alignment},
          _activeKeys{mre, devid, alignment},
          _tableStorage{0, mre, devid},
          _align{alignment} {
      _table.assign(_tableStorage.data());
    }

    HashTable(std::size_t tableSize, memsrc_e mre = memsrc_e::host, ProcID devid = -1,
              std::size_t alignment = 0)
        : _table{buildInstance(mre, devid, next_2pow(tableSize) * reserve_ratio_v)},
          MemoryHandle{mre, devid},
          _tableSize{static_cast<value_t>(next_2pow(tableSize) * reserve_ratio_v)},
          _cnt{1, mre, devid, alignment},
          _activeKeys{next_2pow(tableSize) * reserve_ratio_v, mre, devid, alignment},
          _tableStorage{evaluateTableSize(_tableSize), mre, devid},
          _align{alignment} {
      _table.assign(_tableStorage.data());
    }

    HashTable clone(const MemoryHandle &mh) const {
      HashTable ret{};
      ret.base() = mh;
      ret._tableSize = _tableSize;
      ret._cnt = _cnt.clone(mh);
      ret._activeKeys = _activeKeys.clone(mh);
      ret._tableStorage = _tableStorage.clone(mh);
      ret._table = buildInstance(mh.memspace(), mh.devid(), _tableSize);
      ret._table.assign(ret._tableStorage.data());
      ret._align = _align;
      return ret;
    }
    HashTable(const HashTable &o)
        : _table{buildInstance(o.memspace(), o.devid(), o._tableSize)},
          MemoryHandle{o.base()},
          _tableSize{o._tableSize},
          _cnt{o._cnt},
          _activeKeys{o._activeKeys},
          _tableStorage{o._tableStorage},
          _align{o._align} {
      _table.assign(_tableStorage.data());
    }
    HashTable &operator=(const HashTable &o) {
      if (this == &o) return *this;
      *this = std::move(o.clone(memoryHandle()));
      return *this;
    }
    constexpr HashTable(HashTable &&o) noexcept = default;
    HashTable &operator=(HashTable &&o) noexcept = default;

    inline value_t size() const {
      Vector<value_t> res{1, memsrc_e::host, -1};
      copy({res.base(), res.data()}, {_cnt.base(), _cnt.data()}, sizeof(value_t));
      return res[0];
    }

    value_t _tableSize;
    Vector<value_t> _cnt;
    Vector<key_t> _activeKeys;
    Vector<std::max_align_t> _tableStorage;
    base_t _table;
    std::size_t _align;

  protected:
    constexpr auto evaluateTableSize(value_t capacity) const noexcept {
      using namespace ds;
      uniform_domain<0, Tn, 1, index_seq<0>> dom{wrapv<0>{}, capacity};
      hash_table_snode<key_t, value_t, status_t> node{
          ds::decorations<ds::soa>{}, dom,
          zs::make_tuple(wrapt<key_t>{}, wrapt<value_t>{}, wrapt<status_t>{}), vseq_t<1, 1, 1>{}};
      return snode_size(node) / sizeof(std::max_align_t) + 1;
    }
    constexpr auto buildInstance(memsrc_e mre, ProcID devid, value_t capacity) const noexcept {
      using namespace ds;
      uniform_domain<0, Tn, 1, index_seq<0>> dom{wrapv<0>{}, capacity};
      hash_table_snode<key_t, value_t, status_t> node{
          ds::decorations<ds::soa>{}, dom,
          zs::make_tuple(wrapt<key_t>{}, wrapt<value_t>{}, wrapt<status_t>{}), vseq_t<1, 1, 1>{}};
      auto inst = instance{wrapv<dense>{}, zs::make_tuple(node)};
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
