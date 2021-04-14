#pragma once
#include "Vector.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/math/Vec.h"
#include "zensim/math/bit/Bits.h"
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

  template <typename Tn_, int dim, typename Index, typename Fn> struct HashTable
      : hash_table_instance<vec<std::make_signed_t<Tn_>, dim>, Index, int>,
        MemoryHandle {
    using Tn = std::make_signed_t<Tn_>;
    using hf_t = Fn;
    using key_t = vec<Tn, dim>;
    using value_t = Index;
    using status_t = int;
    using base_t = hash_table_instance<key_t, value_t, status_t>;
    static_assert(
        std::is_convertible_v<decltype(std::declval<Fn>()(std::declval<key_t>())), value_t>,
        "hash function not compatible with hash table key-value type");

    static constexpr value_t sentinel_v{-1};
    static constexpr status_t status_sentinel_v{-1};
    static constexpr std::size_t reserve_ratio_v = 4;

    constexpr auto &self() noexcept { return static_cast<base_t &>(*this); }
    constexpr const auto &self() const noexcept { return static_cast<const base_t &>(*this); }

    HashTable(const Fn &hf, memsrc_e mre = memsrc_e::host, ProcID devid = -1,
              std::size_t alignment = 0)
        : base_t{buildInstance(mre, devid, 0)},
          MemoryHandle{mre, devid},
          _hf{hf},
          _tableSize{0},
          _cnt{mre, devid},
          _activeKeys{mre, devid},
          _align{alignment} {}

    HashTable(const Fn &hf, value_t tableSize, memsrc_e mre = memsrc_e::host, ProcID devid = -1,
              std::size_t alignment = 0)
        : base_t{buildInstance(mre, devid, next_2pow(tableSize) * reserve_ratio_v)},
          MemoryHandle{mre, devid},
          _hf{hf},
          _tableSize{next_2pow(tableSize) * reserve_ratio_v},
          _cnt{1, mre, devid},
          _activeKeys{_tableSize, mre, devid},
          _align{alignment} {}

    hf_t _hf;  ///< hash function
    value_t _tableSize;
    Vector<value_t> _cnt;
    Vector<key_t> _activeKeys;
    std::size_t _align;

  protected:
    constexpr auto buildInstance(memsrc_e mre, ProcID devid, value_t capacity) {
      using namespace ds;
      uniform_domain<0, Tn, 1, index_seq<0>> dom{wrapv<0>{}, capacity};
      hash_table_snode<key_t, value_t, status_t> node{
          ds::decorations<ds::soa>{}, dom,
          zs::make_tuple(wrapt<key_t>{}, wrapt<value_t>{}, wrapt<status_t>{}), vseq_t<1, 1, 1>{}};
      auto inst = instance{wrapv<dense>{}, zs::make_tuple(node)};

      if (capacity) {
        auto memorySource = get_resource_manager().source(mre);
        if (mre == memsrc_e::um) memorySource = memorySource.advisor("PREFERRED_LOCATION", devid);
        /// additional parameters should match allocator_type
        inst.alloc(memorySource);
      }
      return inst;
    }
    constexpr GeneralAllocator getCurrentAllocator() {
      auto memorySource = get_resource_manager().source(this->memspace());
      if (this->memspace() == memsrc_e::um)
        memorySource = memorySource.advisor("PREFERRED_LOCATION", this->devid());
      return memorySource;
    }
  };

  template <typename DomainT, int dim, typename CounterT, typename Fn, typename... Args>
  auto make_hash_table(Fn &&f, Args &&...args) {
    return HashTable<DomainT, dim, CounterT, remove_cvref_t<Fn>>{FWD(f), FWD(args)...};
  }

  template <execspace_e, typename HashTableT, typename = void>
  struct HashTableProxy;  ///< proxy to work within each backends
  template <typename HashTableT> struct HashTableProxy<execspace_e::host, HashTableT> {
    using hf_t = typename HashTableT::hf_t;
    using table_t = typename HashTableT::base_t;
    using key_t = typename HashTableT::key_t;
    using value_t = typename HashTableT::value_t;
    using status_t = typename HashTableT::status_t;

    constexpr HashTableProxy() = default;
    ~HashTableProxy() = default;
    explicit HashTableProxy(HashTableT &table)
        : _hf{table._hf},
          _table{table.self()},
          _tableSize{table._tableSize},
          _cnt{table._cnt.data()},
          _activeKeys{table._activeKeys.data()} {}

    value_t insert(const key_t &key);
    value_t query(const key_t &key) const;

    hf_t _hf;
    table_t _table;
    value_t _tableSize;
    value_t *_cnt;
    key_t *_activeKeys;
  };

  template <execspace_e ExecSpace, typename Tn, int dim, typename Index, typename Fn>
  decltype(auto) proxy(HashTable<Tn, dim, Index, Fn> &table) {
    return HashTableProxy<ExecSpace, HashTable<Tn, dim, Index, Fn>>{table};
  }

}  // namespace zs
