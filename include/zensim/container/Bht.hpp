#pragma once
#include "Bcht.hpp"
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
    static_assert(is_trivially_copyable_v<Tn_>, "Key is not trivially-copyable!");
    static_assert(is_fundamental_v<Tn_>, "Key component should be fundamental!");

    using index_type = zs::make_signed_t<Tn_>;
    using key_type = vec<index_type, dim_>;
    using value_type = zs::make_signed_t<Index>;
    using status_type = int;
    using size_type = zs::make_unsigned_t<Index>;

    static constexpr int dim = dim_;
    static constexpr size_type bucket_size = B;
    static constexpr size_type threshold = bucket_size - 2;
    static_assert(threshold > 0, "bucket load threshold must be positive");

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

    using allocator_type = AllocatorT;
    using difference_type = zs::make_signed_t<size_type>;
    using reference = tuple<key_type &, value_type &>;
    using const_reference = tuple<const key_type &, const value_type &>;
    using pointer = tuple<key_type *, value_type *>;
    using const_pointer = tuple<const key_type *, const value_type *>;

    using hasher_type = universal_hash<key_type>;
    using pair_type = tuple<key_type, value_type>;

    struct Table {
      using storage_key_vector = Vector<storage_key_type, allocator_type>;
      using value_vector = Vector<value_type, allocator_type>;
      using status_vector = Vector<status_type, allocator_type>;
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
      void reset(bool resetIndex = false) {
        keys.reset(0x3f);  // big enough positive integer
        status.reset(-1);  // byte-wise init
        if (resetIndex) indices.reset(-1);
      }

      storage_key_vector keys;
      value_vector indices;
      status_vector status;
    };

    static_assert(
        std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_swap::value,
        "allocator should propagate on copy, move and swap (for impl simplicity)!");

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
      size_t n = next_2pow(entryCnt) * 2;
      return n + (bucket_size - n % bucket_size);
    }
    bht(const allocator_type &allocator, size_t numExpectedEntries)
        : _table{allocator, evaluateTableSize(numExpectedEntries)},
          _tableSize{static_cast<size_type>(evaluateTableSize(numExpectedEntries))},
          _cnt{allocator, 1},
          _activeKeys{allocator, evaluateTableSize(numExpectedEntries)},
          _buildSuccess{allocator, 1} {
      std::mt19937 rng(2);
      // initialize hash funcs
      _hf0 = hasher_type(rng);
      _hf1 = hasher_type(rng);
      _hf2 = hasher_type(rng);

      // _cnt.setVal((value_type)0);
      _cnt.reset(0);
      _table.reset(false);

      _buildSuccess.setVal(1);
    }
    bht(size_t numExpectedEntries, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : bht{get_default_allocator(mre, devid), numExpectedEntries} {}
    bht(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : bht{get_default_allocator(mre, devid), (size_t)0} {}

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
      bht ret{};
      // table
      ret.self().keys = self().keys.clone(allocator);
      ret.self().indices = self().indices.clone(allocator);
      ret.self().status = self().status.clone(allocator);

      ret._cnt = _cnt.clone(allocator);
      ret._tableSize = _tableSize;
      ret._activeKeys = _activeKeys.clone(allocator);
      ret._buildSuccess = _buildSuccess.clone(allocator);
      ret._hf0 = _hf0;
      ret._hf1 = _hf1;
      ret._hf2 = _hf2;
      return ret;
    }
    bht clone(const MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }

    bht(bht &&o) noexcept {
      const bht defaultTable{};
      self() = std::exchange(o.self(), defaultTable.self());
      _cnt = std::exchange(o._cnt, defaultTable._cnt);
      _tableSize = std::exchange(o._tableSize, defaultTable._tableSize);
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
      std::swap(_cnt, o._cnt);
      std::swap(_tableSize, o._tableSize);
      std::swap(_activeKeys, o._activeKeys);
      std::swap(_buildSuccess, o._buildSuccess);
      std::swap(_hf0, o._hf0);
      std::swap(_hf1, o._hf1);
      std::swap(_hf2, o._hf2);
    }
    friend void swap(bht &a, bht &b) noexcept { a.swap(b); }

    size_type size() const { return _cnt.getVal(0); }

#if 0
    template <typename Policy>
    void preserve(Policy &&, size_t numExpectedEntries);
#endif
    template <typename Policy> void reset(Policy &&, bool clearCnt);
    void reset(bool clearCnt) {
      _table.reset(false);
      if (clearCnt) _cnt.reset(0);
      _buildSuccess.setVal(1);
    }

    template <typename Policy> void resize(Policy &&, size_t newCapacity);
    /// @note either scatter or gather
    template <typename Policy, typename MapRange, bool Scatter = false>
    void reorder(Policy &&, MapRange &&mapR, wrapv<Scatter> = {});

    Table _table;
    size_type _tableSize;
    Vector<value_type, allocator_type> _cnt;
    Vector<key_type, allocator_type> _activeKeys;
    Vector<int, allocator_type> _buildSuccess;
    hasher_type _hf0, _hf1, _hf2;
  };

#define ZS_FWD_DECL_BHT_INSTANTIATIONS(CoordIndexType, IndexType, B)                       \
  ZPC_FWD_DECL_TEMPLATE_STRUCT bht<CoordIndexType, 1, IndexType, B, ZSPmrAllocator<>>;     \
  ZPC_FWD_DECL_TEMPLATE_STRUCT bht<CoordIndexType, 2, IndexType, B, ZSPmrAllocator<>>;     \
  ZPC_FWD_DECL_TEMPLATE_STRUCT bht<CoordIndexType, 3, IndexType, B, ZSPmrAllocator<>>;     \
  ZPC_FWD_DECL_TEMPLATE_STRUCT bht<CoordIndexType, 4, IndexType, B, ZSPmrAllocator<>>;     \
  ZPC_FWD_DECL_TEMPLATE_STRUCT bht<CoordIndexType, 1, IndexType, B, ZSPmrAllocator<true>>; \
  ZPC_FWD_DECL_TEMPLATE_STRUCT bht<CoordIndexType, 2, IndexType, B, ZSPmrAllocator<true>>; \
  ZPC_FWD_DECL_TEMPLATE_STRUCT bht<CoordIndexType, 3, IndexType, B, ZSPmrAllocator<true>>; \
  ZPC_FWD_DECL_TEMPLATE_STRUCT bht<CoordIndexType, 4, IndexType, B, ZSPmrAllocator<true>>;

  ZS_FWD_DECL_BHT_INSTANTIATIONS(i32, i32, 16)
  ZS_FWD_DECL_BHT_INSTANTIATIONS(i32, i64, 16)

#if 0
  template <typename HashTableView> struct ResetBHT {
    using hash_table_type = typename HashTableView::hash_table_type;
    explicit ResetBHT(HashTableView tv, bool clearCnt) : table{tv}, clearCnt{clearCnt} {}

    constexpr void operator()(typename HashTableView::size_type entry) noexcept {
      using namespace placeholders;
      table._table.keys[entry] = hash_table_type::key_sentinel_v;
      table._table.indices[entry]
          = hash_table_type::sentinel_v;  // necessary for query to terminate
      table._table.status[entry] = -1;
      if (entry == 0 && clearCnt) *table._cnt = 0;
    }

    HashTableView table;
    bool clearCnt;
  };
#endif

  template <typename Tn, int dim, typename Index, int B, typename Allocator>
  template <typename Policy>
  void bht<Tn, dim, Index, B, Allocator>::reset(Policy &&policy, bool clearCnt) {
#if 0
    constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
    using LsvT = decltype(proxy<space>(*this));
    policy(range(_tableSize), ResetBHT<LsvT>{proxy<space>(*this), clearCnt});
#else
    _table.reset();
    if (clearCnt) _cnt.reset(0);
    _buildSuccess.setVal(1);
#endif
  }

  template <typename TabViewT> struct BhtInsertionOp {
    TabViewT tb;
    constexpr void operator()(typename TabViewT::index_type i) noexcept {
      tb.insert(tb._activeKeys[i], i, false);
    }
  };
  template <typename Tn, int dim, typename Index, int B, typename Allocator>
  template <typename Policy>
  void bht<Tn, dim, Index, B, Allocator>::resize(Policy &&pol, size_t newCapacity) {
    newCapacity = evaluateTableSize(newCapacity);
    if (newCapacity <= _tableSize) return;
    constexpr execspace_e space = RM_REF_T(pol)::exec_tag::value;
    _tableSize = newCapacity;
    // _numBuckets = newCapacity / bucket_size;
    _table.resize(_tableSize);
    reset(false);
    _activeKeys.resize(_tableSize);  // previous records are guaranteed to be preserved
    const auto numEntries = size();
    using TabViewT = decltype(view<space>(*this));
    pol(range(numEntries), BhtInsertionOp<TabViewT>{view<space>(*this)});
  }

  template <typename BhtView, typename KeyView, typename MapIter, bool Scatter> struct ReorderBht {
    using size_type = typename BhtView::size_type;
    using value_type = typename BhtView::value_type;
    using key_type = typename BhtView::key_type;
    explicit ReorderBht(BhtView tb, KeyView orderedKeys, MapIter map)
        : tb{tb}, orderedKeys{orderedKeys}, map{map} {}

    constexpr void operator()(size_type i) {
      size_type j = map[i];
      key_type key{};
      // reorder keys
      if constexpr (Scatter) {  // scatter
        key = tb._activeKeys[i];
        orderedKeys[j] = key;
      } else {  // gather
        key = tb._activeKeys[j];
        orderedKeys[i] = key;
      }
      // remap index table
      auto entry = tb.query(key, false_c);
      if (entry != detail::deduce_numeric_max<value_type>()) {
        if constexpr (Scatter) {
          tb._table.indices[entry] = j;
        } else
          tb._table.indices[entry] = i;
      } else {
        *tb._success = 0;
        // printf("fatal error! should not missing querying an existing key!\n");
      }
    }

    BhtView tb;
    KeyView orderedKeys;
    MapIter map;
  };
  template <typename Tn, int dim, typename Index, int B, typename Allocator>
  template <typename Policy, typename MapRange, bool Scatter>
  void bht<Tn, dim, Index, B, Allocator>::reorder(Policy &&pol, MapRange &&mapR, wrapv<Scatter>) {
    constexpr execspace_e space = RM_REF_T(pol)::exec_tag::value;
    using Ti = RM_CVREF_T(*zs::begin(mapR));
    static_assert(is_integral_v<Ti>,
                  "index mapping range\'s dereferenced type is not an integral.");

    const size_type sz = size();
    if (range_size(mapR) != sz) throw std::runtime_error("index mapping range size mismatch");
    if (!valid_memspace_for_execution(pol, _activeKeys.get_allocator()))
      throw std::runtime_error("current memory location not compatible with the execution policy");

    Vector<key_type, allocator_type> orderedKeys(_activeKeys.get_allocator(),
                                                 _activeKeys.capacity());
    {
      auto tb = view<space>(*this);
      auto oks = view<space>(orderedKeys);
      auto mapIter = zs::begin(mapR);
      pol(range(sz), ReorderBht<RM_CVREF_T(tb), RM_CVREF_T(oks), RM_CVREF_T(mapIter), Scatter>{
                         tb, oks, mapIter});
    }
    _activeKeys = zs::move(orderedKeys);
  }

  /// proxy to work within each backends
  template <execspace_e space_, typename HashTableT, bool Base = false, typename = void>
  struct BHTView {
    static constexpr auto space = space_;
    static constexpr auto exectag = wrapv<space>{};
    static constexpr bool is_const_structure = is_const_v<HashTableT>;
    using hash_table_type = remove_const_t<HashTableT>;
    using hasher_type = typename hash_table_type::hasher_type;

    using storage_key_type = typename hash_table_type::storage_key_type;
    using key_type = typename hash_table_type::key_type;
    using index_type = typename hash_table_type::index_type;
    using value_type = typename hash_table_type::value_type;

    using reference = typename hash_table_type::reference;
    using const_reference = typename hash_table_type::const_reference;
    using pointer = typename hash_table_type::pointer;
    using const_pointer = typename hash_table_type::const_pointer;

    using size_type = typename hash_table_type::size_type;
    using difference_type = typename hash_table_type::difference_type;
    using Tn = typename hash_table_type::index_type;

    using status_type = typename hash_table_type::status_type;

    using unsigned_value_t = make_unsigned_t<value_type>;
    // using unsigned_value_t = conditional_t<sizeof(value_type) == 2, u16,
    //                                       conditional_t<sizeof(value_type) == 4, u32, u64>>;

    static_assert(sizeof(unsigned_value_t) == sizeof(value_type),
                  "unsigned version of value_type not as expected");

    static constexpr int dim = hash_table_type::dim;
    static constexpr size_type bucket_size = hash_table_type::bucket_size;
    static constexpr size_type threshold = hash_table_type::threshold;

    static constexpr auto is_base_c = wrapv<Base>{};
    using storage_key_vector_view_type = decltype(view<space>(
        declval<conditional_t<is_const_structure,
                              const typename hash_table_type::Table::storage_key_vector &,
                              typename hash_table_type::Table::storage_key_vector &>>(),
        is_base_c));
    using value_vector_view_type = decltype(view<space>(
        declval<
            conditional_t<is_const_structure, const typename hash_table_type::Table::value_vector &,
                          typename hash_table_type::Table::value_vector &>>(),
        is_base_c));
    using status_vector_view_type = decltype(view<space>(
        declval<conditional_t<is_const_structure,
                              const typename hash_table_type::Table::status_vector &,
                              typename hash_table_type::Table::status_vector &>>(),
        is_base_c));

    struct table_t {
#if 0
      conditional_t<is_const_structure, const storage_key_type *, storage_key_type *> keys{nullptr};
      conditional_t<is_const_structure, const value_type *, value_type *> indices{nullptr};
      conditional_t<is_const_structure, const status_type *, status_type *> status{nullptr};
#else
      storage_key_vector_view_type keys{};
      value_vector_view_type indices{};
      status_vector_view_type status{};
#endif
    };

    static constexpr auto sentinel_v = hash_table_type::sentinel_v;
    static constexpr auto status_sentinel_v = hash_table_type::status_sentinel_v;
    static constexpr auto failure_token_v = hash_table_type::failure_token_v;

    constexpr BHTView() noexcept = default;
    explicit constexpr BHTView(HashTableT &table)
        : _table{view<space>(table.self().keys, is_base_c, "bht_storage_key"),
                 view<space>(table.self().indices, is_base_c, "bht_indices"),
                 view<space>(table.self().status, is_base_c, "bht_status")},
          _activeKeys{table._activeKeys.data()},
          _tableSize{table._tableSize},
          _cnt{table._cnt.data()},
          _success{table._buildSuccess.data()},
          _numBuckets{table._tableSize / bucket_size},
          _hf0{table._hf0},
          _hf1{table._hf1},
          _hf2{table._hf2} {}

#if defined(__CUDACC__) || defined(__MUSACC__) || defined(__HIPCC__)
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_all<
                  S == execspace_e::cuda || S == execspace_e::musa || S == execspace_e::rocm, !V>
              = 0>
    __forceinline__ __host__ __device__ value_type insert(const key_type &key,
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
            if (localno >= _tableSize - 20) {
              printf("proximity!!! %d -> %d\n", (int)localno, (int)_tableSize);
              *_success = false;
              localno = failure_token_v;
            }
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
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_all<
                  S == execspace_e::cuda || S == execspace_e::musa || S == execspace_e::rocm, !V>
              = 0>
    __forceinline__ __host__ __device__ value_type tile_insert(
        cooperative_groups::thread_block_tile<bucket_size, cooperative_groups::thread_block> &tile,
        const key_type &key, index_type insertion_index = sentinel_v,
        bool enqueueKey = true) noexcept {
#  if 0
      value_type ret;
      if (tile.thread_rank() == 0)
        ret = insert(key, insertion_index, enqueueKey);
      ret = tile.shfl(ret, 0);
      return ret;
#  endif
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
#endif
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_all<is_host_execution<S>(), !V> = 0>
    inline value_type insert(const key_type &key, value_type insertion_index = sentinel_v,
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
            if (localno >= _tableSize - 20) {
              printf("proximity!!! %d -> %d\n", (int)localno, (int)_tableSize);
              *_success = false;
              localno = failure_token_v;
            }
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
          return HashTableT::sentinel_v;
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
        return HashTableT::sentinel_v;
      else
        return detail::deduce_numeric_max<value_type>();
    }
#if defined(__CUDACC__) || defined(__MUSACC__) || defined(__HIPCC__)
    template <
        bool retrieve_index = true, execspace_e S = space,
        enable_if_t<S == execspace_e::cuda || S == execspace_e::musa || S == execspace_e::rocm> = 0>
    __forceinline__ __host__ __device__ value_type tile_query(
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
#endif
    constexpr size_type entry(const key_type &key) const noexcept {
      using namespace placeholders;
      return static_cast<size_type>(query(key, true_c));
    }

    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_t<S == execspace_e::host && !V> = 0>
    void clear() {
      using namespace placeholders;
      // reset counter
      *_cnt = 0;
      // reset table
      for (key_type entry = 0; entry < _tableSize; ++entry) {
        _table.keys[entry] = hash_table_type::key_sentinel_v;
        _table.indices[entry] = HashTableT::sentinel_v;
        _table.status[entry] = HashTableT::status_sentinel_v;
      }
    }

    constexpr auto size() const noexcept { return *_cnt; }

    table_t _table{};
    conditional_t<is_const_structure, const key_type *, key_type *> _activeKeys{nullptr};
    size_type _tableSize{0}, _numBuckets{};  // constness makes non-trivial
    conditional_t<is_const_structure, const value_type *, value_type *> _cnt{nullptr};
    conditional_t<is_const_structure, const int *, int *> _success{nullptr};
    hasher_type _hf0, _hf1, _hf2;

  protected:
    /// @brief helper methods: atomicSwitchIfEqual, atomicLoad
#if defined(__CUDACC__) || defined(__MUSACC__) || defined(__HIPCC__)
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_all<
                  S == execspace_e::cuda || S == execspace_e::musa || S == execspace_e::rocm, !V>
              = 0>
    __forceinline__ __device__ bool atomicSwitchIfEqual(status_type *lock,
                                                        volatile storage_key_type *const dest,
                                                        const storage_key_type &val) noexcept {
      using namespace placeholders;
      constexpr auto key_sentinel_v = hash_table_type::deduce_key_sentinel();
      if constexpr (sizeof(storage_key_type) == 8) {
        static_assert(alignof(storage_key_type) == alignof(u64),
                      "storage key type alignment is not the same as u64");
        const storage_key_type storage_key_sentinel_v = key_sentinel_v;
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

        return (atomic_cas(wrapv<S>{}, const_cast<u64 *>(dst.ptr64), *expected.ptr64,
                           *desired.ptr64)
                << (storage_key_type::num_padded_bytes * 8))
               == (*expected.ptr64 << (storage_key_type::num_padded_bytes * 8));
      } else if constexpr (sizeof(storage_key_type) == 4) {
        static_assert(alignof(storage_key_type) == alignof(u32),
                      "storage key type alignment is not the same as u32");
        const storage_key_type storage_key_sentinel_v = key_sentinel_v;
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

        return (atomic_cas(wrapv<S>{}, const_cast<u32 *>(dst.ptr32), *expected.ptr32,
                           *desired.ptr32)
                << (storage_key_type::num_padded_bytes * 8))
               == (*expected.ptr32 << (storage_key_type::num_padded_bytes * 8));
      }
      /// lock
      while (atomic_exch(wrapv<S>{}, lock, 0) == 0);
      thread_fence(wrapv<S>{});
      /// cas
      storage_key_type temp;
      for (int d = 0; d != dim; ++d) (void)(temp.val(d) = dest->val.data()[d]);
      bool eqn = temp.val == key_sentinel_v;
      if (eqn) {
        for (int d = 0; d != dim; ++d) (void)(dest->val.data()[d] = val.val(d));
      }
      thread_fence(wrapv<S>{});
      /// unlock
      atomic_exch(wrapv<S>{}, lock, HashTableT::status_sentinel_v);
      return eqn;
    }
    /// @ref https://stackoverflow.com/questions/32341081/how-to-have-atomic-load-in-cuda
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_all<
                  S == execspace_e::cuda || S == execspace_e::musa || S == execspace_e::rocm, !V>
              = 0>
    __forceinline__ __device__ key_type
    atomicLoad(status_type *lock, const volatile storage_key_type *const dest) noexcept {
      using namespace placeholders;
      if constexpr (sizeof(storage_key_type) == 8) {
        thread_fence(wrapv<S>{});
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
        // *dst.ptr64 = atomic_or(wrapv<S>{}, const_cast<u64 *>(src.ptr64), (u64)0);
        *dst.ptr64 = *src.ptr64;
        thread_fence(wrapv<S>{});
        return *dst.ptr;
      } else if constexpr (sizeof(storage_key_type) == 4) {
        thread_fence(wrapv<S>{});
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
        // *dst.ptr32 = atomic_or(wrapv<S>{}, const_cast<u32 *>(src.ptr32), (u32)0);
        *dst.ptr32 = *src.ptr32;
        thread_fence(wrapv<S>{});
        return *dst.ptr;
      }
      /// lock
      while (atomic_exch(wrapv<S>{}, lock, 0) == 0);
      thread_fence(wrapv<S>{});
      ///
      key_type return_val;
      for (int d = 0; d != dim; ++d) (void)(return_val.val(d) = dest->val.data()[d]);
      thread_fence(wrapv<S>{});
      /// unlock
      atomic_exch(wrapv<S>{}, lock, HashTableT::status_sentinel_v);
      return return_val;
    }
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_all<
                  S == execspace_e::cuda || S == execspace_e::musa || S == execspace_e::rocm, !V>
              = 0>
    __forceinline__ __device__ key_type atomicTileLoad(
        cooperative_groups::thread_block_tile<bucket_size, cooperative_groups::thread_block> &tile,
        status_type *lock, const volatile storage_key_type *const dest) noexcept {
      using namespace placeholders;
      if constexpr (sizeof(storage_key_type) == 8) {
        thread_fence(wrapv<S>{});
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
        // *dst.ptr64 = atomic_or(wrapv<S>{}, const_cast<u64 *>(src.ptr64), (u64)0);
        *dst.ptr64 = *src.ptr64;
        thread_fence(wrapv<S>{});
        return *dst.ptr;
      } else if constexpr (sizeof(storage_key_type) == 4) {
        thread_fence(wrapv<S>{});
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
        // *dst.ptr32 = atomic_or(wrapv<S>{}, const_cast<u32 *>(src.ptr32), (u32)0);
        *dst.ptr32 = *src.ptr32;
        thread_fence(wrapv<S>{});
        return *dst.ptr;
      }
      /// lock
      if (tile.thread_rank() == 0)
        while (atomic_exch(wrapv<S>{}, lock, 0) == 0);
      tile.sync();
      thread_fence(wrapv<S>{});
      ///
      key_type return_val;
      for (int d = 0; d != dim; ++d) (void)(return_val.val(d) = dest->val.data()[d]);
      thread_fence(wrapv<S>{});
      /// unlock
      if (tile.thread_rank() == 0) atomic_exch(wrapv<S>{}, lock, HashTableT::status_sentinel_v);
      tile.sync();
      return return_val;
    }
#endif
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_all<is_host_execution<S>(), !V> = 0>
    inline bool atomicSwitchIfEqual(status_type *lock, volatile storage_key_type *const dest,
                                    const storage_key_type &val) noexcept {
      constexpr auto execTag = wrapv<S>{};
      using namespace placeholders;
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

        return (atomic_cas(execTag, const_cast<u64 *>(dst.ptr64), *expected.ptr64, *desired.ptr64)
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

        return (atomic_cas(execTag, const_cast<u32 *>(dst.ptr32), *expected.ptr32, *desired.ptr32)
                << (storage_key_type::num_padded_bytes * 8))
               == (*expected.ptr32 << (storage_key_type::num_padded_bytes * 8));
      }
      /// lock
      while (atomic_exch(execTag, lock, 0) == 0);
      thread_fence(execTag);
      /// cas
      storage_key_type temp;  //= volatile_load(dest);
      for (int d = 0; d != dim; ++d) (void)(temp.val(d) = dest->val.data()[d]);
      bool eqn = temp.val == key_sentinel_v;
      if (eqn) {
        for (int d = 0; d != dim; ++d) (void)(dest->val.data()[d] = val.val(d));
      }
      thread_fence(execTag);
      /// unlock
      atomic_exch(execTag, lock, HashTableT::status_sentinel_v);
      return eqn;
    }
    template <execspace_e S = space, bool V = is_const_structure,
              enable_if_all<is_host_execution<S>(), !V> = 0>
    inline key_type atomicLoad(status_type *lock,
                               const volatile storage_key_type *const dest) noexcept {
      constexpr auto execTag = wrapv<S>{};
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
      while (atomic_exch(execTag, lock, 0) == 0);
      thread_fence(execTag);
      ///
      key_type return_val;
      for (int d = 0; d != dim; ++d) (void)(return_val.val(d) = dest->val.data()[d]);
      thread_fence(execTag);
      /// unlock
      atomic_exch(execTag, lock, HashTableT::status_sentinel_v);
      return return_val;
    }
  };

  template <execspace_e ExecSpace, typename Tn, int dim, typename Index, int B, typename Allocator,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(bht<Tn, dim, Index, B, Allocator> &table, wrapv<Base> = {}) {
    return BHTView<ExecSpace, bht<Tn, dim, Index, B, Allocator>, Base>{table};
  }
  template <execspace_e ExecSpace, typename Tn, int dim, typename Index, int B, typename Allocator,
            bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(const bht<Tn, dim, Index, B, Allocator> &table, wrapv<Base> = {}) {
    return BHTView<ExecSpace, const bht<Tn, dim, Index, B, Allocator>, Base>{table};
  }
  template <execspace_e space, typename Tn, int dim, typename Index, int B, typename Allocator>
  decltype(auto) proxy(bht<Tn, dim, Index, B, Allocator> &table) {
    return view<space>(table, false_c);
  }
  template <execspace_e space, typename Tn, int dim, typename Index, int B, typename Allocator>
  decltype(auto) proxy(const bht<Tn, dim, Index, B, Allocator> &table) {
    return view<space>(table, false_c);
  }

#if ZS_ENABLE_SERIALIZATION
  template <typename S, typename Tn, int dim, typename Index, int B>
  void serialize(S &s, bht<Tn, dim, Index, B, ZSPmrAllocator<>> &ht) {
    using C = bht<Tn, dim, Index, B, ZSPmrAllocator<>>;
    if (!ht.memoryLocation().onHost()) {
      ht = ht.clone({memsrc_e::host, -1});
    }

    serialize(s, ht._table.keys);
    serialize(s, ht._table.indices);
    serialize(s, ht._table.status);
    s.template value<sizeof(typename C::size_type)>(ht._tableSize);
    serialize(s, ht._cnt);
    serialize(s, ht._activeKeys);
    serialize(s, ht._buildSuccess);
  }
#endif

}  // namespace zs
