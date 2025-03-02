#pragma once
#include <random>

#include "Vector.hpp"
#include "zensim/execution/Atomics.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/execution/Intrinsics.hpp"
#include "zensim/math/Hash.hpp"
#include "zensim/math/Vec.h"
#include "zensim/math/bit/Bits.h"
#include "zensim/math/probability/Random.hpp"
#include "zensim/memory/MemoryResource.h"
#include "zensim/py_interop/HashUtils.hpp"
#include "zensim/resource/Resource.h"
#include "zensim/types/Iterator.h"
#if defined(__CUDACC__) || defined(__MUSACC__) || defined(__HIPCC__)
#  include <cooperative_groups.h>
#endif
#if ZS_ENABLE_OPENMP
#  include "zensim/omp/Omp.h"
#endif

namespace zs {

  /// ref: https://github.com/owensgroup/BGHT
  // ref: Better GPU Hash Tables
  // Muhammad A. Awad, Saman Ashkiani, Serban D. Porumbescu, Martín Farach-Colton, and John D. Owens
  template <typename KeyT> struct universal_hash : universal_hash_base<KeyT> {
    using base_t = universal_hash_base<KeyT>;
    using base_t::_hashx;
    using base_t::_hashy;
    using key_type = KeyT;
    using result_type = u32;

    // borrowed from cudpp
    static constexpr u32 prime_divisor = 4294967291u;

    universal_hash() noexcept : base_t{call_random(1), call_random(2)} {}
    universal_hash(std::mt19937 &rng) {
      _hashx = rng() % prime_divisor;
      if (_hashx < 1) _hashx = 1;
      _hashy = rng() % prime_divisor;
    }
    constexpr universal_hash(u32 hash_x, u32 hash_y) : base_t{hash_x, hash_y} {}
    ~universal_hash() = default;
    universal_hash(const universal_hash &) = default;
    universal_hash(universal_hash &&) = default;
    universal_hash &operator=(const universal_hash &) = default;
    universal_hash &operator=(universal_hash &&) = default;
  };

  template <typename KeyT> struct vec_pack_hash {
    // usually used with no-original-key-compare hash table
    using key_type = KeyT;
    using result_type = u64;

    vec_pack_hash() noexcept : _hashx{call_random(1)}, _hashy{call_random(2)} {}
    vec_pack_hash(std::mt19937 &rng) : _hashx{0}, _hashy{0} {
      _hashx = rng() % universal_hash<key_type>::prime_divisor;
      if (_hashx < 1) _hashx = 1;
      _hashy = rng() % universal_hash<key_type>::prime_divisor;
    }
    constexpr vec_pack_hash(u32 hash_x, u32 hash_y) : _hashx(hash_x), _hashy(hash_y) {}
    ~vec_pack_hash() = default;
    vec_pack_hash(const vec_pack_hash &) = default;
    vec_pack_hash(vec_pack_hash &&) = default;
    vec_pack_hash &operator=(const vec_pack_hash &) = default;
    vec_pack_hash &operator=(vec_pack_hash &&) = default;

    template <bool isVec = is_vec<key_type>::value, enable_if_t<!isVec> = 0>
    constexpr result_type operator()(const key_type &key) const noexcept {
      return (((_hashx ^ key) + _hashy) % universal_hash<key_type>::prime_divisor);
    }
    template <bool isVec = is_vec<key_type>::value, enable_if_t<isVec> = 0>
    constexpr result_type operator()(const key_type &key) const noexcept {
      if constexpr (is_integral_v<typename key_type::value_type>) {
        if constexpr (sizeof(typename key_type::value_type) == 4) {  // i32, u32
          // this is the most frequently used case
          // only preserve low 21-bit index of each dimension
          if constexpr (key_type::extent == 3) {
            auto extract = [](typename key_type::value_type val) -> u64 {
              if constexpr (is_unsigned_v<typename key_type::value_type>)
                return (u64)val & (u64)0x1fffffu;
              else if constexpr (is_signed_v<typename key_type::value_type>)
                return ((u64)val & (u64)0xfffffu) | ((u64)(val < 0 ? 1 : 0) << (u64)20);
            };
            return (extract(key.val(0)) << (u64)42) | (extract(key.val(1)) << (u64)21)
                   | extract(key.val(2));
          } else if constexpr (key_type::extent == 2)
            return ((u64)key.val(0) << (u64)32) | (u64)key.val(1);
          else if constexpr (key_type::extent == 1)
            return (u64)key.val(0);
        } else if constexpr (sizeof(typename key_type::value_type) == 2) {  // i16, u16
          if constexpr (key_type::extent <= 4) {
            u64 ret{0};
            for (int i = 0; i != key_type::extent; ++i) ret = (ret << 16) | (u64)key.val(i);
            return ret;
          }
        } else if constexpr (sizeof(typename key_type::value_type) == 1) {  // i8, (u)char
          if constexpr (key_type::extent <= 8) {
            u64 ret{0};
            for (int i = 0; i != key_type::extent; ++i) ret = (ret << 8) | (u64)key.val(i);
            return ret;
          }
        }
      }
      // default routine
      return (result_type)universal_hash<key_type>{_hashx, _hashy}(key);
    }

    u32 _hashx;
    u32 _hashy;
  };

  // directly compare key to avoid duplication
  template <typename KeyT, typename Index = int, bool KeyCompare = true,
            typename HashT = universal_hash<KeyT>, int B = 16,
            typename AllocatorT = ZSPmrAllocator<>>
  struct bcht {
    static_assert(is_zs_allocator<AllocatorT>::value,
                  "Hashtable only works with zspmrallocator for now.");
    static_assert(is_same_v<KeyT, remove_cvref_t<KeyT>>, "Key is not cvref-unqualified type!");
    static_assert(std::is_default_constructible_v<KeyT>, "Key is not default-constructible!");
    static_assert(std::is_trivially_copyable_v<KeyT>, "Key is not trivially-copyable!");

    static constexpr auto bucket_size = B;
    static constexpr bool compare_key = KeyCompare;

    using hasher_type = HashT;
    using allocator_type = AllocatorT;
    static constexpr bool key_is_vec = is_vec<KeyT>::value;

    using original_key_type = KeyT;
    template <bool KeyIsVec> static constexpr auto deduce_key_type() noexcept {
      if constexpr (KeyIsVec)
        // avoid storage issues arising from other fancy zs::VecInterface types
        return wrapt<zs::vec<typename KeyT::value_type, KeyT::extent>>{};
      else
        return wrapt<KeyT>{};
    }
    using key_type = typename decltype(deduce_key_type<key_is_vec>())::type;
    using hashed_key_type = RM_CVREF_T(declval<hasher_type>()(declval<key_type>()));
    using mapped_hashed_key_type
        = conditional_t<sizeof(hashed_key_type) == 8, u64,
                        conditional_t<sizeof(hashed_key_type) == 4, u32,
                                      conditional_t<sizeof(hashed_key_type) == 4, u16, void>>>;
    static_assert(is_unsigned_v<mapped_hashed_key_type>,
                  "hashed key type should be an unsigned integer");
    using storage_key_type = conditional_t<compare_key, key_type, mapped_hashed_key_type>;
    //
    using index_type = zs::make_signed_t<Index>;
    using value_type = zs::tuple<key_type, index_type>;  // use original 'key_type' here
    using size_type = zs::make_unsigned_t<Index>;
    using difference_type = index_type;
    using reference = zs::tuple<storage_key_type &, index_type &>;
    using const_reference = zs::tuple<const storage_key_type &, const index_type &>;
    using pointer = zs::tuple<storage_key_type *, index_type *>;
    using const_pointer = zs::tuple<const storage_key_type *, const index_type *>;

    template <bool KeyIsVec> static constexpr key_type deduce_key_sentinel() noexcept {
      if constexpr (KeyIsVec) {
        typename key_type::value_type v{0};
        for (int i = 0; i != sizeof(typename key_type::value_type); ++i) v = (v << 8) | 0x3f;
        return key_type::constant(v);
      } else {
        key_type v{0};
        for (int i = 0; i != sizeof(key_type); ++i) v = (v << 8) | 0x3f;
        return v;
      }
    }
    static constexpr key_type key_sentinel_v = deduce_key_sentinel<key_is_vec>();
    static constexpr int deduce_dimension() noexcept {
      if constexpr (key_is_vec)
        return key_type::extent;
      else
        return 1;
    }
    static constexpr int dim = deduce_dimension();

    static constexpr mapped_hashed_key_type deduce_hkey_sentinel() noexcept {
      mapped_hashed_key_type v{0};
      for (int i = 0; i != sizeof(mapped_hashed_key_type); ++i) v = (v << 8) | 0x3f;
      return v;
    }
    static constexpr mapped_hashed_key_type hkey_sentinel_v = deduce_hkey_sentinel();

    static constexpr auto deduce_compare_key_sentinel() noexcept {
      if constexpr (compare_key)
        return key_sentinel_v;
      else
        return hkey_sentinel_v;
    }
    static constexpr storage_key_type compare_key_sentinel_v = deduce_compare_key_sentinel();
    static constexpr index_type sentinel_v = -1;
    static constexpr index_type failure_token_v = detail::deduce_numeric_max<index_type>();

    static size_t padded_capacity(size_t capacity) noexcept {
      if (auto remainder = capacity % bucket_size; remainder) capacity += (bucket_size - remainder);
      return capacity;
    }

    struct Table {
      Table() = default;
      Table(const Table &) = default;
      Table(Table &&) noexcept = default;
      Table &operator=(const Table &) = default;
      Table &operator=(Table &&) noexcept = default;
      Table(const allocator_type &allocator, size_t capacity)
          : keys{allocator, capacity},
            indices{allocator, capacity},
            status{allocator, capacity / bucket_size} {}

      void resize(size_t newCap) {
        keys.resize(newCap);
        indices.resize(newCap);
        status.resize(newCap / bucket_size);
      }
      void swap(Table &o) noexcept {
        std::swap(keys, o.keys);
        std::swap(indices, o.indices);
        std::swap(status, o.status);
      }
      void reset(bool resetIndex = false) {
        keys.reset(0x3f);  // big enough positive integer
        status.reset(-1);  // byte-wise init
        if (resetIndex) indices.reset(-1);
      }
      friend void swap(Table &a, Table &b) { a.swap(b); }

      Table clone(const allocator_type &allocator) const {
        Table ret{};
        ret.keys = keys.clone(allocator);
        ret.indices = indices.clone(allocator);
        ret.status = status.clone(allocator);
        return ret;
      }

      Vector<storage_key_type, allocator_type> keys;
      Vector<index_type, allocator_type> indices;
      // used as locks for critical section during insertion
      Vector<int, allocator_type> status;
    };

    static_assert(
        std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value
            && std::allocator_traits<allocator_type>::propagate_on_container_swap::value,
        "allocator should propagate on copy, move and swap (for impl simplicity)!");

    struct equal_to {
      template <typename T> constexpr bool operator()(const T &lhs, const T &rhs) const {
        return lhs == rhs;
      }
      // more specialized than the above
      template <typename T0, typename T1>
      constexpr bool operator()(const VecInterface<T0> &lhs, const VecInterface<T1> &rhs) const {
        return lhs == rhs;
      }
    };

    constexpr decltype(auto) memoryLocation() const noexcept { return _cnt.memoryLocation(); }
    constexpr ProcID devid() const noexcept { return memoryLocation().devid(); }
    constexpr memsrc_e memspace() const noexcept { return memoryLocation().memspace(); }
    decltype(auto) get_allocator() const noexcept { return _cnt.get_allocator(); }
    decltype(auto) get_default_allocator(memsrc_e mre, ProcID devid) const {
      if constexpr (is_virtual_zs_allocator<allocator_type>::value)
        return get_virtual_memory_source(mre, devid, (size_t)1 << (size_t)36, "STACK");
      else
        return get_memory_source(mre, devid);
    }

    inline static u32 deduce_num_chains(size_t cap) {
      // maximum number of cuckoo chains
      double lg_input_size = (float)(std::log((double)cap) / std::log(2.0));
      constexpr unsigned max_iter_const = 7;
      return static_cast<u32>(max_iter_const * lg_input_size);
    }

    bcht(const allocator_type &allocator, size_t capacity)
        : _capacity{padded_capacity(capacity) * 2},
          _numBuckets{(size_type)_capacity / bucket_size},
          _table{allocator, _capacity},
          _activeKeys{allocator, _capacity},
          _buildSuccess{allocator, 1},
          _cnt{allocator, 1},
          _maxCuckooChains{0},
          _hf0{},
          _hf1{},
          _hf2{} {
      _buildSuccess.setVal((index_type)0);

      _cnt.setVal((size_type)0);
      _table.reset(false);

      _maxCuckooChains = deduce_num_chains(_capacity);

      std::mt19937 rng(2);
      // initialize hash funcs
      _hf0 = hasher_type(rng);
      _hf1 = hasher_type(rng);
      _hf2 = hasher_type(rng);

      // mark complete status
      _buildSuccess.setVal(1);
    }
    bcht(size_t capacity, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : bcht{get_default_allocator(mre, devid), capacity} {}
    bcht(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : bcht{get_default_allocator(mre, devid), (size_t)0} {}

    ~bcht() = default;

    void swap(bcht &o) noexcept {
      std::swap(_capacity, o._capacity);
      std::swap(_numBuckets, o._numBuckets);
      std::swap(_table, o._table);
      std::swap(_activeKeys, o._activeKeys);
      std::swap(_buildSuccess, o._buildSuccess);
      std::swap(_cnt, o._cnt);
      std::swap(_maxCuckooChains, o._maxCuckooChains);
      std::swap(_hf0, o._hf0);
      std::swap(_hf1, o._hf1);
      std::swap(_hf2, o._hf2);
    }
    friend void swap(bcht &a, bcht &b) { a.swap(b); }

    bcht(const bcht &o)
        : _capacity{o._capacity},
          _numBuckets{o._numBuckets},
          _table{o._table},
          _activeKeys{o._activeKeys},
          _buildSuccess{o._buildSuccess},
          _cnt{o._cnt},
          _maxCuckooChains{o._maxCuckooChains},
          _hf0{o._hf0},
          _hf1{o._hf1},
          _hf2{o._hf2} {}
    bcht &operator=(const bcht &o) {
      if (this == &o) return *this;
      bcht tmp(o);
      swap(tmp);
      return *this;
    }

    bcht(bcht &&o) noexcept {
      const bcht defaultTable{};
      _capacity = std::exchange(o._capacity, defaultTable._capacity);
      _numBuckets = std::exchange(o._numBuckets, defaultTable._numBuckets);
      _table = std::exchange(o._table, defaultTable._table);
      _activeKeys = std::exchange(o._activeKeys, defaultTable._activeKeys);
      _buildSuccess = std::exchange(o._buildSuccess, defaultTable._buildSuccess);
      _cnt = std::exchange(o._cnt, defaultTable._cnt);
      _maxCuckooChains = std::exchange(o._maxCuckooChains, defaultTable._maxCuckooChains);
      _hf0 = std::exchange(o._hf0, defaultTable._hf0);
      _hf1 = std::exchange(o._hf1, defaultTable._hf1);
      _hf2 = std::exchange(o._hf2, defaultTable._hf2);
    }
    bcht &operator=(bcht &&o) noexcept {
      if (this == &o) return *this;
      bcht tmp(std::move(o));
      swap(tmp);
      return *this;
    }

    bcht clone(const allocator_type &allocator) const {
      bcht ret{allocator, 0};
      ret._capacity = _capacity;
      ret._numBuckets = _numBuckets;
      if (_capacity) {
        ret._table = _table.clone(allocator);
        ret._activeKeys = _activeKeys.clone(allocator);
        ret._buildSuccess = _buildSuccess.clone(allocator);
        ret._cnt = _cnt.clone(allocator);
      }
      ret._maxCuckooChains = _maxCuckooChains;
      ret._hf0 = _hf0;
      ret._hf1 = _hf1;
      ret._hf2 = _hf2;
      return ret;
    }
    bcht clone(const MemoryLocation &mloc) const {
      return clone(get_default_allocator(mloc.memspace(), mloc.devid()));
    }

    inline size_type size() const { return _cnt.getVal(0); }

    void reset(bool clearCnt) {
      _buildSuccess.setVal(0);
      _table.keys.reset(0x3f);
      // no need to worry about clearing indices
      _table.status.reset(-1);
      if (clearCnt) _cnt.setVal(0);
      _buildSuccess.setVal(1);
    }

    template <typename Policy> void resize(Policy &&, size_t newCapacity);

    size_t _capacity;  // make sure this comes ahead
    size_type _numBuckets;
    Table _table;
    zs::Vector<key_type> _activeKeys;
    zs::Vector<u8> _buildSuccess;
    zs::Vector<size_type> _cnt;
    u32 _maxCuckooChains;
    hasher_type _hf0, _hf1, _hf2;
  };

  template <typename KeyT, typename IndexT, bool KeyCompare, typename HashT, int B,
            typename AllocatorT>
  template <typename Policy>
  void bcht<KeyT, IndexT, KeyCompare, HashT, B, AllocatorT>::resize(Policy &&pol,
                                                                    size_t newCapacity) {
    newCapacity = padded_capacity(newCapacity) * 2;
    if (newCapacity <= _capacity) return;
    constexpr execspace_e space = RM_REF_T(pol)::exec_tag::value;
    _capacity = newCapacity;
    _numBuckets = _capacity / bucket_size;
    _table.resize(_capacity);
    reset(false);
    _activeKeys.resize(_capacity);  // previous records are guaranteed to be preserved
    _maxCuckooChains = deduce_num_chains(_capacity);
    const auto numEntries = size();
    pol(range(numEntries), [tb = proxy<space>(*this)] ZS_LAMBDA(index_type i) mutable {
      tb.insert(tb._activeKeys[i], i, false);
    });
  }

  struct mars_rng_32 {
    u32 y;
    constexpr mars_rng_32() : y(2463534242) {}
    constexpr u32 operator()() {
      y ^= (y << 13);
      y = (y >> 17);
      return (y ^= (y << 5));
    }
  };

  template <execspace_e space, typename BCHT, bool Base = false, typename = void> struct BCHTView {
    static constexpr bool is_const_structure = is_const_v<BCHT>;
    using hash_table_type = remove_const_t<BCHT>;
    static constexpr auto exectag = wrapv<space>{};

    static constexpr auto bucket_size = hash_table_type::bucket_size;
    static constexpr bool compare_key = hash_table_type::compare_key;
    static constexpr int spin_iter_cap = 32;

    using allocator_type = typename hash_table_type::allocator_type;
    using hasher_type = typename hash_table_type::hasher_type;
    using original_key_type = typename hash_table_type::original_key_type;
    using key_type = typename hash_table_type::key_type;
    using hashed_key_type = typename hash_table_type::hashed_key_type;
    using mapped_hashed_key_type = typename hash_table_type::mapped_hashed_key_type;
    using storage_key_type = typename hash_table_type::storage_key_type;

    using equal_to = typename hash_table_type::equal_to;

    using index_type = typename hash_table_type::index_type;
    using value_type = typename hash_table_type::value_type;
    using size_type = typename hash_table_type::size_type;
    using difference_type = typename hash_table_type::difference_type;
    using pointer = typename hash_table_type::pointer;
    using const_pointer = typename hash_table_type::const_pointer;
    using reference = typename hash_table_type::reference;
    using const_reference = typename hash_table_type::const_reference;

    static constexpr bool key_is_vec = hash_table_type::key_is_vec;
    static constexpr int dim = hash_table_type::dim;
    static constexpr key_type key_sentinel_v = hash_table_type::key_sentinel_v;
    static constexpr mapped_hashed_key_type hkey_sentinel_v = hash_table_type::hkey_sentinel_v;
    static constexpr storage_key_type compare_key_sentinel_v
        = hash_table_type::compare_key_sentinel_v;
    static constexpr index_type sentinel_v = hash_table_type::sentinel_v;
    static constexpr index_type failure_token_v = hash_table_type::failure_token_v;

    struct table_t {
#if 0
      conditional_t<is_const_structure, const storage_key_type *, storage_key_type *> keys{nullptr};
      conditional_t<is_const_structure, const index_type *, index_type *> indices{nullptr};
      conditional_t<is_const_structure, const int *, int *> status{nullptr};
#else
      conditional_t<is_const_structure,
                    VectorView<space, const Vector<storage_key_type, allocator_type>, Base>,
                    VectorView<space, Vector<storage_key_type, allocator_type>, Base>>
          keys{};
      conditional_t<is_const_structure,
                    VectorView<space, const Vector<index_type, allocator_type>, Base>,
                    VectorView<space, Vector<index_type, allocator_type>, Base>>
          indices{};
      conditional_t<is_const_structure, VectorView<space, const Vector<int, allocator_type>, Base>,
                    VectorView<space, Vector<int, allocator_type>, Base>>
          status{};
#endif
    };

    BCHTView() noexcept = default;
    explicit constexpr BCHTView(BCHT &table)
        : _table{view<space>(table._table.keys, wrapv<Base>{}),
                 view<space>(table._table.indices, wrapv<Base>{}),
                 view<space>(table._table.status, wrapv<Base>{})},
          _activeKeys{view<space>(table._activeKeys, wrapv<Base>{})},
          _cnt{table._cnt.data()},
          _success{table._buildSuccess.data()},
          _numBuckets{table._numBuckets},
          _maxCuckooChains{table._maxCuckooChains},
          _hf0{table._hf0},
          _hf1{table._hf1},
          _hf2{table._hf2} {}

    constexpr size_t capacity() const noexcept { return (size_t)_numBuckets * (size_t)bucket_size; }
    constexpr auto transKey(const key_type &key) const noexcept {
      if constexpr (compare_key)
        return key;
      else
        return reinterpret_bits<mapped_hashed_key_type>(_hf0(key));
    }

#if defined(__CUDACC__) || defined(__MUSACC__) || defined(__HIPCC__)

    /// helper construct
    struct CoalescedGroup : cooperative_groups::coalesced_group {
      using base_t = cooperative_groups::coalesced_group;

      __host__ __device__ CoalescedGroup(const cooperative_groups::coalesced_group &group) noexcept
          : base_t{group} {}
      ~CoalescedGroup() = default;

      __forceinline__ __host__ __device__ unsigned long long num_threads() const {
#  if __CUDA_ARCH__
        return static_cast<const base_t *>(this)->num_threads();
#  else
        return ~(unsigned long long)0;
#  endif
      }
      __forceinline__ __host__ __device__ unsigned long long thread_rank() const {
#  if __CUDA_ARCH__
        return static_cast<const base_t *>(this)->thread_rank();
#  else
        return ~(unsigned long long)0;
#  endif
      }
      __forceinline__ __host__ __device__ unsigned ballot(int pred) const {
#  if __CUDA_ARCH__
        return static_cast<const base_t *>(this)->ballot(pred);
#  else
        return ~(unsigned)0;
#  endif
      }
      /// @note ref:
      /// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
      template <typename T, enable_if_t<std::is_fundamental_v<T>> = 0>
      __forceinline__ __host__ __device__ T shfl(T var, int srcLane) const {
#  if __CUDA_ARCH__
        return static_cast<const base_t *>(this)->shfl(var, srcLane);
#  else
        return 0;
#  endif
      }
      template <typename VecT, enable_if_t<is_vec<VecT>::value> = 0>
      __forceinline__ __host__ __device__ VecT shfl(const VecT &var, int srcLane) const {
#  if __CUDA_ARCH__
        VecT ret{};
        for (typename VecT::index_type i = 0; i != VecT::extent; ++i)
          ret.val(i) = this->shfl(var.val(i), srcLane);
        return ret;
#  else
        return VecT::zeros();
#  endif
      }
      __forceinline__ __host__ __device__ int ffs(int x) const {
#  if __CUDA_ARCH__
        return ::__ffs(x);
#  else
        return -1;
#  endif
      }
      __forceinline__ __host__ __device__ void thread_fence() const {
#  if __CUDA_ARCH__
        ::__threadfence();  // device-scope
#  else
#  endif
      }
    };

    ///
    /// insertion
    // @enqueue_key: whether pushing inserted key-index pair into _activeKeys
    ///
    template <execspace_e S = space, enable_if_all<S == execspace_e::cuda || S == execspace_e::musa
                                                   || S == execspace_e::rocm>
                                     = 0>
    __forceinline__ __device__ index_type tile_insert(
        cooperative_groups::thread_block_tile<bucket_size, cooperative_groups::thread_block> &tile,
        const original_key_type &key, index_type insertion_index = sentinel_v,
        bool enqueueKey = true) noexcept {
      namespace cg = ::cooperative_groups;
      if (_numBuckets == 0) return failure_token_v;
      constexpr auto compare_key_sentinel_v = hash_table_type::deduce_compare_key_sentinel();

      const int cap = __popc(tile.ballot(1));  // assume active pattern 0...001111 [15, 14, ..., 0]

      mars_rng_32 rng;
      u32 cuckoo_counter = 0;
      // auto bucket_id = reinterpret_bits<mapped_hashed_key_type>(_hf0(key)) % _numBuckets;
      auto bucket_offset
          = reinterpret_bits<mapped_hashed_key_type>(_hf0(key)) % _numBuckets * bucket_size;
      auto lane_id = tile.thread_rank();
      auto load_key = [&bucket_offset, &keys = _table.keys](index_type i) -> storage_key_type {
        volatile storage_key_type *key_dst
            = const_cast<volatile storage_key_type *>(&keys[bucket_offset + i]);
        if constexpr (compare_key && key_is_vec) {
          storage_key_type ret{};
          for (typename original_key_type::index_type i = 0; i != original_key_type::extent; ++i)
            ret.val(i) = key_dst->data()[i];
          return ret;
        } else
          return *key_dst;
      };

      index_type no = insertion_index;

      storage_key_type insertion_key = transKey(key);
      int spin_iter = 0;
      do {
#  if 1
        int exist = 0;
        for (int i = lane_id; i < bucket_size; i += cap)
          if (equal_to{}(insertion_key, load_key(i))) {
            exist = 1;
            break;
          }
        for (int stride = 1; stride < cap; stride <<= 1) {
          int tmp = tile.shfl(exist, lane_id + stride);
          if (lane_id + stride < cap) exist |= tmp;
        }
        exist = tile.shfl(exist, 0);
        if (exist) return sentinel_v;
#  else
        // simultaneously check every slot in the bucket
        storage_key_type lane_key = _table.keys[bucket_offset + lane_id];
        // if the key already exist, exit early
        // even though it may be moved out of this bucket to another
        if (tile.any(equal_to{}(lane_key, insertion_key))) return sentinel_v;
#  endif

#  if 1
        int load = 0;
        for (int i = lane_id; i < bucket_size; i += cap)
          if (!equal_to{}(compare_key_sentinel_v, load_key(i))) ++load;
        for (int stride = 1; stride < cap; stride <<= 1) {
          int tmp = tile.shfl(load, lane_id + stride);
          if (lane_id + stride < cap) load += tmp;
        }
        load = tile.shfl(load, 0);
#  else
        // if another tile just inserted the same key during this period,
        // then it shall fail the following insertion procedure
        // compute load
        auto load_bitmap = tile.ballot(!equal_to{}(lane_key, compare_key_sentinel_v));
        int load = __popc(load_bitmap);
#  endif

        // if bucket is not full
        if (load != bucket_size) {
          bool casSuccess = false;
          // then only the elected lane is atomically inserting
          if (lane_id == 0) {
            spin_iter = 0;
            while (atomic_cas(wrapv<S>{}, &_table.status[bucket_offset / bucket_size], -1, 0) != -1
                   && ++spin_iter != spin_iter_cap);  // acquiring spin lock
          }
          if (spin_iter = tile.shfl(spin_iter, 0); spin_iter == spin_iter_cap) {
            // check again
            continue;
          }
          thread_fence(wrapv<S>{});
#  if 1
          exist = 0;
          for (int i = lane_id; i < bucket_size; i += cap)
            if (equal_to{}(insertion_key, load_key(i))) {
              exist = 1;
              break;
            }
          for (int stride = 1; stride < cap; stride <<= 1) {
            int tmp = tile.shfl(exist, lane_id + stride);
            if (lane_id + stride < cap) exist |= tmp;
          }
          exist = tile.shfl(exist, 0);  // implicit tile sync here
          if (exist) {
            if (lane_id == 0)  // release lock
              atomic_exch(wrapv<S>{}, &_table.status[bucket_offset / bucket_size], -1);
            return sentinel_v;
          }
#  else
          // read again after locking the bucket
          lane_key = _table.keys[bucket_offset + lane_id];
          if (tile.any(equal_to{}(lane_key, insertion_key))) {  // implicit tile sync here
            if (lane_id == 0)                                   // release lock
              atomic_exch(wrapv<S>{}, &_table.status[bucket_offset / bucket_size], -1);
            return sentinel_v;  // although key found, return sentinel to suggest doing nothing
          }
#  endif

          if (lane_id == 0) {
            if constexpr (compare_key) {
              // storage_key_type retrieved_val = *const_cast<storage_key_type *>(key_dst);
              storage_key_type retrieved_val = load_key(load);
              if (equal_to{}(retrieved_val,
                             compare_key_sentinel_v)) {  // this slot not yet occupied
                volatile storage_key_type *key_dst
                    = const_cast<volatile storage_key_type *>(&_table.keys[bucket_offset + load]);
                if constexpr (key_is_vec) {
                  for (typename original_key_type::index_type i = 0; i != original_key_type::extent;
                       ++i)
                    key_dst->data()[i] = insertion_key.val(i);
                } else
                  *key_dst = insertion_key;
                casSuccess = true;
              }
            } else {
              casSuccess = atomic_cas(wrapv<S>{}, &_table.keys[bucket_offset + load],
                                      compare_key_sentinel_v, insertion_key)
                           == compare_key_sentinel_v;
            }

            if (casSuccess) {  // process index as well
              if (insertion_index == sentinel_v) {
                no = atomic_add(wrapv<S>{}, _cnt, (size_type)1);
                insertion_index = no;
              }
#  if 1
              *const_cast<volatile index_type *>(&_table.indices[bucket_offset + load])
                  = insertion_index;
#  else
              atomic_exch(wrapv<S>{}, &_table.indices[bucket_offset + load], insertion_index);
#  endif
            }
            thread_fence(wrapv<S>{});
            atomic_exch(wrapv<S>{}, &_table.status[bucket_offset / bucket_size], -1);
          }

          // may fail due to the slot taken by another insertion
          // or fail due to existence check
          casSuccess = tile.shfl(casSuccess, 0);
          if (casSuccess) {
            no = tile.shfl(no, 0);
            if (lane_id == 0 && enqueueKey) _activeKeys[no] = key;
            return no;
          }
        } else {
          if (lane_id == 0) {
            spin_iter = 0;
            while (atomic_cas(wrapv<S>{}, &_table.status[bucket_offset / bucket_size], -1, 0) != -1
                   && ++spin_iter != spin_iter_cap);  // acquiring spin lock
          }
          if (spin_iter = tile.shfl(spin_iter, 0); spin_iter == spin_iter_cap) {
            // check again
            continue;
          }
          if (lane_id == 0) {
            auto random_location = rng() % bucket_size;
            thread_fence(wrapv<S>{});
            volatile storage_key_type *key_dst = const_cast<volatile storage_key_type *>(
                &_table.keys[bucket_offset + random_location]);
            auto old_key = *const_cast<storage_key_type *>(key_dst);
            if constexpr (compare_key && key_is_vec) {
              for (typename original_key_type::index_type i = 0; i != original_key_type::extent;
                   ++i)
                key_dst->data()[i] = insertion_key.val(i);
            } else
              *key_dst = insertion_key;
            // _table.keys[bucket_offset + random_location] = insertion_key;

            if (insertion_index == sentinel_v) {
              no = atomic_add(wrapv<S>{}, _cnt, (size_type)1);
              insertion_index = no;
              // casSuccess = true; //not finished yet, evicted key reinsertion is success
            }
            auto old_index = atomic_exch(
                wrapv<S>{}, &_table.indices[bucket_offset + random_location], insertion_index);
            // volatile index_type *index_dst = const_cast<volatile index_type *>(
            //    &_table.indices[bucket_offset + random_location]);
            // auto old_index = *const_cast<index_type *>(index_dst);
            //*index_dst = insertion_index;
            // _table.indices[bucket_offset + random_location] = insertion_index;

            thread_fence(wrapv<S>{});
            atomic_exch(wrapv<S>{}, &_table.status[bucket_offset / bucket_size], -1);
            // should be old keys instead, not (h)ashed keys
            auto bucket0 = reinterpret_bits<mapped_hashed_key_type>(_hf0(old_key)) % _numBuckets;
            auto bucket1 = reinterpret_bits<mapped_hashed_key_type>(_hf1(old_key)) % _numBuckets;
            auto bucket2 = reinterpret_bits<mapped_hashed_key_type>(_hf2(old_key)) % _numBuckets;

            auto new_bucket_id = bucket0;
            new_bucket_id = bucket_offset == bucket1 * bucket_size ? bucket2 : new_bucket_id;
            new_bucket_id = bucket_offset == bucket0 * bucket_size ? bucket1 : new_bucket_id;

            bucket_offset = new_bucket_id * bucket_size;

            insertion_key = old_key;
            insertion_index = old_index;
          }
          bucket_offset = tile.shfl(bucket_offset, 0);  // implicit tile sync
          cuckoo_counter++;
        }
      } while (cuckoo_counter < _maxCuckooChains);
      return failure_token_v;
    }

    template <execspace_e S = space, enable_if_all<S == execspace_e::cuda || S == execspace_e::musa
                                                   || S == execspace_e::rocm>
                                     = 0>
    __forceinline__ __host__ __device__ index_type
    group_insert(CoalescedGroup &tile, const original_key_type &key,
                 index_type insertion_index = sentinel_v, bool enqueueKey = true) noexcept {
      namespace cg = ::cooperative_groups;
      if (_numBuckets == 0) return failure_token_v;
      constexpr auto compare_key_sentinel_v = hash_table_type::deduce_compare_key_sentinel();

      const int cap = math::min((int)tile.size(), (int)bucket_size);
      const int syncCap = math::max((int)tile.size(), (int)bucket_size);

      mars_rng_32 rng;
      u32 cuckoo_counter = 0;
      // auto bucket_id = reinterpret_bits<mapped_hashed_key_type>(_hf0(key)) % _numBuckets;
      auto bucket_offset
          = reinterpret_bits<mapped_hashed_key_type>(_hf0(key)) % _numBuckets * bucket_size;
      auto lane_id = tile.thread_rank();
      auto load_key = [&bucket_offset, &keys = _table.keys](index_type i) -> storage_key_type {
        volatile storage_key_type *key_dst
            = const_cast<volatile storage_key_type *>(&keys[bucket_offset + i]);
        if constexpr (compare_key && key_is_vec) {
          storage_key_type ret{};
          for (typename original_key_type::index_type i = 0; i != original_key_type::extent; ++i)
            ret.val(i) = key_dst->data()[i];
          return ret;
        } else
          return *key_dst;
      };

      index_type no = insertion_index;

      storage_key_type insertion_key = transKey(key);
      int spin_iter = 0;
      do {
        int exist = 0;
        for (int i = lane_id; i < bucket_size; i += cap)
          if (equal_to{}(insertion_key, load_key(i))) {
            exist = 1;
            break;
          }
        for (int stride = 1; stride < syncCap; stride <<= 1) {
          int tmp = tile.shfl(exist, lane_id + stride);
          if (lane_id + stride < cap) exist |= tmp;
        }
        exist = tile.shfl(exist, 0);
        if (exist) return sentinel_v;

        int load = 0;
        for (int i = lane_id; i < bucket_size; i += cap)
          if (!equal_to{}(compare_key_sentinel_v, load_key(i))) ++load;
        for (int stride = 1; stride < syncCap; stride <<= 1) {
          int tmp = tile.shfl(load, lane_id + stride);
          if (lane_id + stride < cap) load += tmp;
        }
        load = tile.shfl(load, 0);

        // if bucket is not full
        if (load != bucket_size) {
          bool casSuccess = false;
          // then only the elected lane is atomically inserting
          if (lane_id == 0) {
            spin_iter = 0;
            while (atomic_cas(wrapv<S>{}, &_table.status[bucket_offset / bucket_size], -1, 0) != -1
                   && ++spin_iter != spin_iter_cap);  // acquiring spin lock
          }
          if (spin_iter = tile.shfl(spin_iter, 0); spin_iter == spin_iter_cap) {
            // check again
            continue;
          }
          tile.thread_fence();

          exist = 0;
          for (int i = lane_id; i < bucket_size; i += cap)
            if (equal_to{}(insertion_key, load_key(i))) {
              exist = 1;
              break;
            }
          for (int stride = 1; stride < syncCap; stride <<= 1) {
            int tmp = tile.shfl(exist, lane_id + stride);
            if (lane_id + stride < cap) exist |= tmp;
          }
          exist = tile.shfl(exist, 0);  // implicit tile sync here
          if (exist) {
            if (lane_id == 0)  // release lock
              atomic_exch(wrapv<S>{}, &_table.status[bucket_offset / bucket_size], -1);
            return sentinel_v;
          }

          if (lane_id == 0) {
            if constexpr (compare_key) {
              // storage_key_type retrieved_val = *const_cast<storage_key_type *>(key_dst);
              storage_key_type retrieved_val = load_key(load);
              if (equal_to{}(retrieved_val,
                             compare_key_sentinel_v)) {  // this slot not yet occupied
                volatile storage_key_type *key_dst = const_cast<volatile storage_key_type *>(
                    &_table.keys[bucket_offset + (u32)load]);
                if constexpr (key_is_vec) {
                  for (typename original_key_type::index_type i = 0; i != original_key_type::extent;
                       ++i)
                    key_dst->data()[i] = insertion_key.val(i);
                } else
                  *key_dst = insertion_key;
                casSuccess = true;
              }
            } else {
              casSuccess = atomic_cas(wrapv<S>{}, &_table.keys[bucket_offset + load],
                                      compare_key_sentinel_v, insertion_key)
                           == compare_key_sentinel_v;
            }

            if (casSuccess) {  // process index as well
              if (insertion_index == sentinel_v) {
                no = atomic_add(wrapv<S>{}, _cnt, (size_type)1);
                insertion_index = no;
              }
#  if 1
              *const_cast<volatile index_type *>(&_table.indices[bucket_offset + (u32)load])
                  = insertion_index;
#  else
              atomic_exch(wrapv<S>{}, &_table.indices[bucket_offset + load], insertion_index);
#  endif
            }
            tile.thread_fence();
            atomic_exch(wrapv<S>{}, &_table.status[bucket_offset / bucket_size], -1);
          }

          // may fail due to the slot taken by another insertion
          // or fail due to existence check
          casSuccess = tile.shfl(casSuccess, 0);
          if (casSuccess) {
            no = tile.shfl(no, 0);
            if (lane_id == 0 && enqueueKey) _activeKeys[no] = key;
            return no;
          }
        } else {
          if (lane_id == 0) {
            spin_iter = 0;
            while (atomic_cas(wrapv<S>{}, &_table.status[bucket_offset / bucket_size], -1, 0) != -1
                   && ++spin_iter != spin_iter_cap);  // acquiring spin lock
          }
          if (spin_iter = tile.shfl(spin_iter, 0); spin_iter == spin_iter_cap) {
            // check again
            continue;
          }
          if (lane_id == 0) {
            auto random_location = rng() % bucket_size;
            tile.thread_fence();
            volatile storage_key_type *key_dst = const_cast<volatile storage_key_type *>(
                &_table.keys[bucket_offset + random_location]);
            auto old_key = *const_cast<storage_key_type *>(key_dst);
            if constexpr (compare_key && key_is_vec) {
              for (typename original_key_type::index_type i = 0; i != original_key_type::extent;
                   ++i)
                key_dst->data()[i] = insertion_key.val(i);
            } else
              *key_dst = insertion_key;
            // _table.keys[bucket_offset + random_location] = insertion_key;

            if (insertion_index == sentinel_v) {
              no = atomic_add(wrapv<S>{}, _cnt, (size_type)1);
              insertion_index = no;
              // casSuccess = true; //not finished yet, evicted key reinsertion is success
            }
            auto old_index = atomic_exch(
                wrapv<S>{}, &_table.indices[bucket_offset + random_location], insertion_index);
            // volatile index_type *index_dst = const_cast<volatile index_type *>(
            //    &_table.indices[bucket_offset + random_location]);
            // auto old_index = *const_cast<index_type *>(index_dst);
            //*index_dst = insertion_index;
            // _table.indices[bucket_offset + random_location] = insertion_index;

            tile.thread_fence();
            atomic_exch(wrapv<S>{}, &_table.status[bucket_offset / bucket_size], -1);
            // should be old keys instead, not (h)ashed keys
            auto bucket0 = reinterpret_bits<mapped_hashed_key_type>(_hf0(old_key)) % _numBuckets;
            auto bucket1 = reinterpret_bits<mapped_hashed_key_type>(_hf1(old_key)) % _numBuckets;
            auto bucket2 = reinterpret_bits<mapped_hashed_key_type>(_hf2(old_key)) % _numBuckets;

            auto new_bucket_id = bucket0;
            new_bucket_id = bucket_offset == bucket1 * bucket_size ? bucket2 : new_bucket_id;
            new_bucket_id = bucket_offset == bucket0 * bucket_size ? bucket1 : new_bucket_id;

            bucket_offset = new_bucket_id * bucket_size;

            insertion_key = old_key;
            insertion_index = old_index;
          }
          bucket_offset = tile.shfl(bucket_offset, 0);  // implicit tile sync
          cuckoo_counter++;
        }
      } while (cuckoo_counter < _maxCuckooChains);
      return failure_token_v;
    }

#  if 1
    template <execspace_e S = space, enable_if_all<S == execspace_e::cuda || S == execspace_e::musa
                                                   || S == execspace_e::rocm>
                                     = 0>
    [[maybe_unused]] __forceinline__ __host__ __device__ index_type
    insert(const original_key_type &insertion_key, index_type insertion_index = sentinel_v,
           bool enqueueKey = true,
           CoalescedGroup group = cooperative_groups::coalesced_threads()) noexcept {
      namespace cg = ::cooperative_groups;

      bool has_work = true;  // is this visible to rest threads in tile cuz of __forceinline__ ??
      bool success = true;
      index_type result = sentinel_v;

      u32 work_queue = group.ballot(has_work);
      while (work_queue) {
        auto cur_rank = group.ffs(work_queue) - 1;
        auto cur_work = group.shfl(insertion_key, cur_rank);
        auto cur_index = group.shfl(insertion_index, cur_rank);  // gather index as well
        auto id = group_insert(group, cur_work, cur_index, enqueueKey);

        if (group.thread_rank() == cur_rank) {
          result = id;
          success = id != failure_token_v;
          has_work = false;
        }
        work_queue = group.ballot(has_work);
      }

      if (!group.all(success)) *_success = false;
      return result;
    }
#  else
    /// use this simplified hash insertion
    template <execspace_e S = space, enable_if_all<S == execspace_e::cuda || S == execspace_e::musa
                                                   || S == execspace_e::rocm>
                                     = 0>
    [[maybe_unused]] __forceinline__ __host__ __device__ index_type
    insert(const original_key_type &insertion_key_, index_type insertion_index = sentinel_v,
           bool enqueueKey = true) noexcept {
      if (_numBuckets == 0) return failure_token_v;

      storage_key_type insertion_key = transKey(insertion_key_);
      auto bucket_offset = reinterpret_bits<mapped_hashed_key_type>(_hf0(insertion_key))
                           % _numBuckets * bucket_size;
      auto res = tryInsertKeyValue(bucket_offset, insertion_key, insertion_index);
      if (res == failure_token_v)
        *_success = false;
      else if (res != sentinel_v && enqueueKey)
        _activeKeys[res] = insertion_key_;
      return res;
    }
#  endif

    /// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
    template <typename T, enable_if_t<std::is_fundamental_v<T>> = 0>
    __forceinline__ __host__ __device__ T tile_shfl(
        cooperative_groups::thread_block_tile<bucket_size, cooperative_groups::thread_block> &tile,
        T var, int srcLane) const {
#  if __CUDA_ARCH__
      return tile.shfl(var, srcLane);
#  else
      return 0;
#  endif
    }
    template <typename VecT, enable_if_t<is_vec<VecT>::value> = 0>
    __forceinline__ __host__ __device__ VecT tile_shfl(
        cooperative_groups::thread_block_tile<bucket_size, cooperative_groups::thread_block> &tile,
        const VecT &var, int srcLane) const {
#  if __CUDA_ARCH__
      VecT ret{};
      for (typename VecT::index_type i = 0; i != VecT::extent; ++i)
        ret.val(i) = tile_shfl(tile, var.val(i), srcLane);
      return ret;
#  else
      return VecT::zeros();
#  endif
    }
    template <execspace_e S = space, enable_if_all<S == execspace_e::cuda || S == execspace_e::musa
                                                   || S == execspace_e::rocm>
                                     = 0>
    [[maybe_unused]] __forceinline__ __device__ index_type insertUnsafe(
        const original_key_type &insertion_key, index_type insertion_index = sentinel_v,
        bool enqueueKey = true,
        cooperative_groups::thread_block_tile<bucket_size, cooperative_groups::thread_block> tile
        = cooperative_groups::tiled_partition<bucket_size>(
            cooperative_groups::this_thread_block())) noexcept {
      namespace cg = ::cooperative_groups;

      bool has_work = true;  // is this visible to rest threads in tile cuz of __forceinline__ ??
      bool success = true;
      index_type result = sentinel_v;

      u32 work_queue = tile.ballot(has_work);
      while (work_queue) {
        auto cur_rank = __ffs(work_queue) - 1;
        auto cur_work = tile_shfl(tile, insertion_key, cur_rank);
        auto cur_index = tile.shfl(insertion_index, cur_rank);  // gather index as well
        auto id = tile_insert(tile, cur_work, cur_index, enqueueKey);

        if (tile.thread_rank() == cur_rank) {
          result = id;
          success = id != failure_token_v;
          has_work = false;
        }
        work_queue = tile.ballot(has_work);
      }

      if (!tile.all(success)) *_success = false;
      return result;
    }

    ///
    /// query
    ///
    // https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
    template <
        bool retrieve_index = true, execspace_e S = space,
        enable_if_all<S == execspace_e::cuda || S == execspace_e::musa || S == execspace_e::rocm>
        = 0>
    __forceinline__ __host__ __device__ index_type tile_query(
        cooperative_groups::thread_block_tile<bucket_size, cooperative_groups::thread_block> &tile,
        const original_key_type &key, wrapv<retrieve_index> = {}) const noexcept {
      namespace cg = ::cooperative_groups;
      if (_numBuckets == 0) {
        if constexpr (retrieve_index)
          return sentinel_v;
        else
          return detail::deduce_numeric_max<index_type>();
      }
      constexpr auto compare_key_sentinel_v = hash_table_type::deduce_compare_key_sentinel();

      const int cap = __popc(tile.ballot(1));  // assume active pattern 0...001111 [15, 14, ..., 0]
      // printf("rank[%d] tile size: %d, mask: %x, %d active\n", tile.thread_rank(),
      //       tile.num_threads(), tile.ballot(1), cap);

      auto bucket_offset
          = reinterpret_bits<mapped_hashed_key_type>(_hf0(key)) % _numBuckets * bucket_size;
      auto lane_id = tile.thread_rank();

      storage_key_type query_key = transKey(key);

      for (int iter = 0; iter < 3; ++iter) {
#  if 1
        // cg::reduce requires compute capability 8.0+
        int location = bucket_size;
        for (int i = lane_id; i < bucket_size; i += cap)
          if (equal_to{}(query_key, _table.keys[bucket_offset + i])) {
            location = i;
            break;
          }
        for (int stride = 1; stride < cap; stride <<= 1) {
          int tmp = tile.shfl(location, lane_id + stride);
          if (lane_id + stride < cap) location = location < tmp ? location : tmp;
        }
        location = tile.shfl(location, 0);
        if (location != bucket_size) {
          if constexpr (retrieve_index) {
            index_type found_value = _table.indices[bucket_offset + location];
            return found_value;
          } else {
            return bucket_offset + location;
          }
        }
#  else
        storage_key_type lane_key = _table.keys[bucket_offset + lane_id];
        auto key_exist_bitmap = tile.ballot(equal_to{}(query_key, lane_key));
        int key_lane = __ffs(key_exist_bitmap);
        int location = key_lane - 1;
        if (location != -1) {
          if constexpr (retrieve_index) {
            index_type found_value = _table.indices[bucket_offset + location];
            return found_value;
          } else {
            return bucket_offset + location;
          }
        }
#  endif
        // check empty slots
        // failed not because the bucket is full, but because there is none
        else {
#  if 1
          int load = 0;
          for (int i = lane_id; i < bucket_size; i += cap)
            if (!equal_to{}(compare_key_sentinel_v, _table.keys[bucket_offset + i])) ++load;
          for (int stride = 1; stride < cap; stride <<= 1) {
            int tmp = tile.shfl(load, lane_id + stride);
            if (lane_id + stride < cap) load += tmp;
          }
          load = tile.shfl(load, 0);
          if (load < bucket_size) return sentinel_v;
#  else
          if (__popc(tile.ballot(!equal_to{}(lane_key, compare_key_sentinel_v))) < bucket_size)
            return sentinel_v;
#  endif
          else
            bucket_offset = iter == 0 ? reinterpret_bits<mapped_hashed_key_type>(_hf1(key))
                                            % _numBuckets * bucket_size
                                      : reinterpret_bits<mapped_hashed_key_type>(_hf2(key))
                                            % _numBuckets * bucket_size;
        }
      }
      if constexpr (retrieve_index)
        return sentinel_v;
      else
        return detail::deduce_numeric_max<index_type>();
    }

    template <
        bool retrieve_index = true, execspace_e S = space,
        enable_if_all<S == execspace_e::cuda || S == execspace_e::musa || S == execspace_e::rocm>
        = 0>
    __forceinline__ __host__ __device__ index_type group_query(CoalescedGroup &tile,
                                                               const original_key_type &key,
                                                               wrapv<retrieve_index>
                                                               = {}) const noexcept {
      if (_numBuckets == 0) {
        if constexpr (retrieve_index)
          return sentinel_v;
        else
          return detail::deduce_numeric_max<index_type>();
      }
      constexpr auto compare_key_sentinel_v = hash_table_type::deduce_compare_key_sentinel();
      const int cap = math::min((int)tile.size(), (int)bucket_size);
      const int syncCap = math::max((int)tile.size(), (int)bucket_size);
      auto bucket_offset
          = reinterpret_bits<mapped_hashed_key_type>(_hf0(key)) % _numBuckets * bucket_size;
      auto lane_id = tile.thread_rank();
      storage_key_type query_key = transKey(key);
      for (int iter = 0; iter < 3; ++iter) {
        // cg::reduce requires compute capability 8.0+
        int location = bucket_size;
        for (int i = lane_id; i < bucket_size; i += cap)
          if (equal_to{}(query_key, _table.keys[bucket_offset + i])) {
            location = i;
            break;
          }
        for (int stride = 1; stride < syncCap; stride <<= 1) {
          int tmp = tile.shfl(location, lane_id + stride);
          if (lane_id + stride < cap) location = location < tmp ? location : tmp;
        }
        location = tile.shfl(location, 0);
        if (location != bucket_size) {
          if constexpr (retrieve_index) {
            index_type found_value = _table.indices[bucket_offset + location];
            return found_value;
          } else {
            return bucket_offset + location;
          }
        }
        // check empty slots
        // failed not because the bucket is full, but because there is none
        else {
          int load = 0;
          for (int i = lane_id; i < bucket_size; i += cap)
            if (!equal_to{}(compare_key_sentinel_v, _table.keys[bucket_offset + i])) ++load;
          for (int stride = 1; stride < syncCap; stride <<= 1) {
            int tmp = tile.shfl(load, lane_id + stride);
            if (lane_id + stride < cap) load += tmp;
          }
          load = tile.shfl(load, 0);
          if (load < bucket_size)
            return sentinel_v;
          else
            bucket_offset = iter == 0 ? reinterpret_bits<mapped_hashed_key_type>(_hf1(key))
                                            % _numBuckets * bucket_size
                                      : reinterpret_bits<mapped_hashed_key_type>(_hf2(key))
                                            % _numBuckets * bucket_size;
        }
      }
      if constexpr (retrieve_index)
        return sentinel_v;
      else
        return detail::deduce_numeric_max<index_type>();
    }

    template <
        bool retrieve_index = true, execspace_e S = space,
        enable_if_all<S == execspace_e::cuda || S == execspace_e::musa || S == execspace_e::rocm>
        = 0>
    __forceinline__ __host__ __device__ index_type single_query(const original_key_type &key,
                                                                wrapv<retrieve_index>
                                                                = {}) const noexcept {
      if (_numBuckets == 0) {
        if constexpr (retrieve_index)
          return sentinel_v;
        else
          return detail::deduce_numeric_max<index_type>();
      }
      constexpr auto compare_key_sentinel_v = hash_table_type::deduce_compare_key_sentinel();
      auto bucket_offset
          = reinterpret_bits<mapped_hashed_key_type>(_hf0(key)) % _numBuckets * bucket_size;
      storage_key_type query_key = transKey(key);
      for (int iter = 0; iter < 3; ++iter) {
        // cg::reduce requires compute capability 8.0+
        int location = bucket_size;
        for (int i = 0; i < bucket_size; ++i)
          if (equal_to{}(query_key, _table.keys[bucket_offset + i])) {
            location = i;
            break;
          }
        if (location != bucket_size) {
          if constexpr (retrieve_index) {
            index_type found_value = _table.indices[bucket_offset + location];
            return found_value;
          } else {
            return bucket_offset + location;
          }
        }
        // check empty slots
        // failed not because the bucket is full, but because there is none
        else {
          int load = 0;
          for (int i = 0; i < bucket_size; ++i)
            if (!equal_to{}(compare_key_sentinel_v, _table.keys[bucket_offset + i])) ++load;
          if (load < bucket_size)
            return sentinel_v;
          else
            bucket_offset = iter == 0 ? reinterpret_bits<mapped_hashed_key_type>(_hf1(key))
                                            % _numBuckets * bucket_size
                                      : reinterpret_bits<mapped_hashed_key_type>(_hf2(key))
                                            % _numBuckets * bucket_size;
        }
      }
      if constexpr (retrieve_index)
        return sentinel_v;
      else
        return detail::deduce_numeric_max<index_type>();
    }

    template <execspace_e S = space, enable_if_all<S == execspace_e::cuda || S == execspace_e::musa
                                                   || S == execspace_e::rocm>
                                     = 0>
    [[nodiscard]] __forceinline__ __host__ __device__ index_type
    query(const original_key_type &find_key,
          CoalescedGroup tile = cooperative_groups::coalesced_threads()) const noexcept {
      bool has_work = true;  // is this visible to rest threads in tile cuz of __forceinline__ ??
      index_type result = sentinel_v;
      auto work_queue = tile.ballot(has_work);
      while (work_queue) {
        auto cur_rank = tile.ffs(work_queue) - 1;
        auto cur_work = tile.shfl(find_key, cur_rank);
        auto find_result = group_query(tile, cur_work);
        if (tile.thread_rank() == cur_rank) {
          result = find_result;
          has_work = false;
        }
        work_queue = tile.ballot(has_work);
      }
      return result;
    }

    template <execspace_e S = space, enable_if_all<S == execspace_e::cuda || S == execspace_e::musa
                                                   || S == execspace_e::rocm>
                                     = 0>
    [[nodiscard]] __forceinline__ __host__ __device__ index_type queryUnsafe(
        const original_key_type &find_key,
        cooperative_groups::thread_block_tile<bucket_size, cooperative_groups::thread_block> tile
        = cooperative_groups::tiled_partition<bucket_size>(cooperative_groups::this_thread_block()))
        const noexcept {
      namespace cg = ::cooperative_groups;

      bool has_work = true;  // is this visible to rest threads in tile cuz of __forceinline__ ??
      index_type result = sentinel_v;

      auto work_queue = tile.ballot(has_work);
      while (work_queue) {
        auto cur_rank = ffs(wrapv<S>{}, work_queue) - 1;
        auto cur_work = tile_shfl(tile, find_key, cur_rank);
        auto find_result = tile_query(tile, cur_work);

        if (tile.thread_rank() == cur_rank) {
          result = find_result;
          has_work = false;
        }
        work_queue = tile.ballot(has_work);
      }
      return result;
    }

    ///
    /// entry (return the location of the key)
    ///
    template <execspace_e S = space, enable_if_all<S == execspace_e::cuda || S == execspace_e::musa
                                                   || S == execspace_e::rocm>
                                     = 0>
    [[nodiscard]] __forceinline__ __host__ __device__ index_type
    entry(const original_key_type &find_key,
          cooperative_groups::thread_block_tile<bucket_size, cooperative_groups::thread_block> tile
          = cooperative_groups::tiled_partition<bucket_size>(
              cooperative_groups::this_thread_block())) const noexcept {
      namespace cg = ::cooperative_groups;

      bool has_work = true;  // is this visible to rest threads in tile cuz of __forceinline__ ??
      index_type result = sentinel_v;

      auto work_queue = tile.ballot(has_work);
      while (work_queue) {
        auto cur_rank = ffs(wrapv<S>{}, work_queue) - 1;
        auto cur_work = tile.shfl(find_key, cur_rank);
        auto find_result = tile_query(tile, cur_work, wrapv<false>{});

        if (tile.thread_rank() == cur_rank) {
          result = find_result;
          has_work = false;
        }
        work_queue = tile.ballot(has_work);
      }
      return result;
    }
#endif

    template <execspace_e S = space, enable_if_all<is_host_execution<S>()> = 0>
    [[maybe_unused]] inline index_type insert(const original_key_type &key,
                                              index_type insertion_index = sentinel_v,
                                              const bool enqueueKey = true) noexcept {
      if (_numBuckets == 0) return failure_token_v;
      mars_rng_32 rng;
      u32 cuckoo_counter = 0;
      auto bucket_offset
          = reinterpret_bits<mapped_hashed_key_type>(_hf0(key)) % _numBuckets * bucket_size;
      auto load_key = [&bucket_offset, &keys = _table.keys](index_type i) -> storage_key_type {
        return keys[bucket_offset + i];
      };

      index_type no = insertion_index;

      storage_key_type insertion_key = transKey(key);

      do {
        bool exist = false;
        for (int i = 0; i != bucket_size; ++i)
          if (equal_to{}(insertion_key, load_key(i))) exist = true;
        if (exist) return sentinel_v;

        int load = 0;
        for (int i = 0; i != bucket_size; ++i)
          if (!equal_to{}(compare_key_sentinel_v, load_key(i))) ++load;

        // if bucket is not full
        if (load != bucket_size) {
          bool casSuccess = false;
          // critical section
          {
            // check duplication again
            if constexpr (compare_key) {
              volatile storage_key_type *key_dst
                  = const_cast<volatile storage_key_type *>(&_table.keys[bucket_offset + load]);
              storage_key_type retrieved_val = load_key(load);
              if (equal_to{}(retrieved_val,
                             compare_key_sentinel_v)) {  // this slot not yet occupied
                if constexpr (key_is_vec) {
                  for (typename original_key_type::index_type i = 0; i != original_key_type::extent;
                       ++i)
                    key_dst->data()[i] = insertion_key.val(i);
                } else
                  *key_dst = insertion_key;
                casSuccess = true;
              }
            } else {
              casSuccess = atomic_cas(seq_c, &_table.keys[bucket_offset + load],
                                      compare_key_sentinel_v, insertion_key)
                           == compare_key_sentinel_v;
            }

            if (casSuccess) {  // process index as well
              if (insertion_index == sentinel_v) {
                no = atomic_add(seq_c, _cnt, (size_type)1);
                insertion_index = no;
              }
              _table.indices[bucket_offset + load] = insertion_index;
            }
          }

          if (casSuccess) {
            if (enqueueKey) _activeKeys[no] = key;
            return no;
          }
        } else {
          {
            auto random_location = rng() % bucket_size;
            // thread_fence(wrapv<S>{});
            volatile storage_key_type *key_dst = const_cast<volatile storage_key_type *>(
                &_table.keys[bucket_offset + random_location]);
            auto old_key = *const_cast<storage_key_type *>(key_dst);
            if constexpr (compare_key && key_is_vec) {
              for (typename original_key_type::index_type i = 0; i != original_key_type::extent;
                   ++i)
                key_dst->data()[i] = insertion_key.val(i);
            } else
              *key_dst = insertion_key;

            if (insertion_index == sentinel_v) {
              no = atomic_add(seq_c, _cnt, (size_type)1);
              insertion_index = no;
            }
            auto old_index = atomic_exch(seq_c, &_table.indices[bucket_offset + random_location],
                                         insertion_index);

            // thread_fence(seq_c);
            // should be old keys instead, not (h)ashed keys
            auto bucket0 = reinterpret_bits<mapped_hashed_key_type>(_hf0(old_key)) % _numBuckets;
            auto bucket1 = reinterpret_bits<mapped_hashed_key_type>(_hf1(old_key)) % _numBuckets;
            auto bucket2 = reinterpret_bits<mapped_hashed_key_type>(_hf2(old_key)) % _numBuckets;

            auto new_bucket_id = bucket0;
            new_bucket_id = bucket_offset == bucket1 * bucket_size ? bucket2 : new_bucket_id;
            new_bucket_id = bucket_offset == bucket0 * bucket_size ? bucket1 : new_bucket_id;

            bucket_offset = new_bucket_id * bucket_size;

            insertion_key = old_key;
            insertion_index = old_index;
          }
          cuckoo_counter++;
        }
      } while (cuckoo_counter < _maxCuckooChains);
      return failure_token_v;
    }

    template <bool retrieve_index = true, execspace_e S = space,
              enable_if_all<S == execspace_e::host> = 0>
    [[nodiscard]] inline index_type query(const original_key_type &key,
                                          wrapv<retrieve_index> = {}) const noexcept {
      if (_numBuckets == 0) {
        if constexpr (retrieve_index)
          return sentinel_v;
        else
          return detail::deduce_numeric_max<index_type>();
      }
      auto bucket_offset
          = reinterpret_bits<mapped_hashed_key_type>(_hf0(key)) % _numBuckets * bucket_size;
      storage_key_type query_key = transKey(key);
      for (int iter = 0; iter < 3; ++iter) {
        int location = -1;
        for (int i = 0; i != bucket_size; ++i)
          if (equal_to{}(query_key, _table.keys[bucket_offset + i])) {
            location = i;
            break;
          }
        if (location != -1) {
          if constexpr (retrieve_index) {
            index_type found_value = _table.indices[bucket_offset + location];
            return found_value;
          } else
            return bucket_offset + location;
        } else {
          int load = 0;
          for (int i = 0; i != bucket_size; ++i)
            if (!equal_to{}(compare_key_sentinel_v, _table.keys[bucket_offset + i])) ++load;

          if (load < bucket_size)
            return sentinel_v;
          else
            bucket_offset = iter == 0 ? reinterpret_bits<mapped_hashed_key_type>(_hf1(key))
                                            % _numBuckets * bucket_size
                                      : reinterpret_bits<mapped_hashed_key_type>(_hf2(key))
                                            % _numBuckets * bucket_size;
        }
      }
      if constexpr (retrieve_index)
        return sentinel_v;
      else
        return detail::deduce_numeric_max<index_type>();
    }

    ///
    /// entry (return the location of the key)
    ///
    template <execspace_e S = space, enable_if_all<S == execspace_e::host> = 0>
    [[nodiscard]] inline index_type entry(const original_key_type &find_key) const noexcept {
      return query(find_key, false_c);
    }

#if ZS_ENABLE_OPENMP
    template <execspace_e S = space, enable_if_all<S == execspace_e::openmp> = 0>
    [[maybe_unused]] inline index_type insert(const original_key_type &key,
                                              index_type insertion_index = sentinel_v,
                                              const bool enqueueKey = true) noexcept {
      if (_numBuckets == 0) return failure_token_v;
      mars_rng_32 rng;
      u32 cuckoo_counter = 0;
      auto bucket_offset
          = reinterpret_bits<mapped_hashed_key_type>(_hf0(key)) % _numBuckets * bucket_size;
      auto load_key = [&bucket_offset, &keys = _table.keys](index_type i) -> storage_key_type {
        volatile storage_key_type *key_dst
            = const_cast<volatile storage_key_type *>(&keys[bucket_offset + i]);
        if constexpr (compare_key && key_is_vec) {
          storage_key_type ret{};
          for (typename original_key_type::index_type i = 0; i != original_key_type::extent; ++i)
            ret.val(i) = key_dst->data()[i];
          return ret;
        } else
          return *key_dst;
      };

      index_type no = insertion_index;

      storage_key_type insertion_key = transKey(key);
      int spin_iter = 0;

      do {
        bool exist = false;
        for (int i = 0; i != bucket_size; ++i)
          if (equal_to{}(insertion_key, load_key(i))) exist = true;
        if (exist) return sentinel_v;

        int load = 0;
        for (int i = 0; i != bucket_size; ++i)
          if (!equal_to{}(compare_key_sentinel_v, load_key(i))) ++load;

        // if bucket is not full
        if (load != bucket_size) {
          bool casSuccess = false;

          spin_iter = 0;
          while (atomic_cas(omp_c, &_table.status[bucket_offset / bucket_size], -1, 0) != -1
                 && ++spin_iter != spin_iter_cap);
          if (spin_iter == spin_iter_cap) continue;
          thread_fence(omp_c);

          exist = false;
          for (int i = 0; i != bucket_size; ++i)
            if (equal_to{}(insertion_key, load_key(i))) exist = true;
          if (exist) {
            atomic_exch(omp_c, &_table.status[bucket_offset / bucket_size], -1);
            return sentinel_v;
          }

          {
            // check duplication again
            if constexpr (compare_key) {
              storage_key_type retrieved_val = load_key(load);
              if (equal_to{}(retrieved_val,
                             compare_key_sentinel_v)) {  // this slot not yet occupied
                volatile storage_key_type *key_dst
                    = const_cast<volatile storage_key_type *>(&_table.keys[bucket_offset + load]);
                if constexpr (key_is_vec) {
                  for (typename original_key_type::index_type i = 0; i != original_key_type::extent;
                       ++i)
                    key_dst->data()[i] = insertion_key.val(i);
                } else
                  *key_dst = insertion_key;
                casSuccess = true;
              }
            } else {
              casSuccess = atomic_cas(omp_c, &_table.keys[bucket_offset + load],
                                      compare_key_sentinel_v, insertion_key)
                           == compare_key_sentinel_v;
            }

            if (casSuccess) {  // process index as well
              if (insertion_index == sentinel_v) {
                no = atomic_add(omp_c, _cnt, (size_type)1);
                insertion_index = no;
              }
              *const_cast<volatile index_type *>(&_table.indices[bucket_offset + load])
                  = insertion_index;
            }
            thread_fence(omp_c);
            atomic_exch(omp_c, &_table.status[bucket_offset / bucket_size], -1);
          }

          if (casSuccess) {
            if (enqueueKey) _activeKeys[no] = key;
            return no;
          }
        } else {
          {
            spin_iter = 0;
            while (atomic_cas(omp_c, &_table.status[bucket_offset / bucket_size], -1, 0) != -1
                   && ++spin_iter != spin_iter_cap);
            if (spin_iter == spin_iter_cap) continue;

            auto random_location = rng() % bucket_size;
            thread_fence(omp_c);
            volatile storage_key_type *key_dst = const_cast<volatile storage_key_type *>(
                &_table.keys[bucket_offset + random_location]);
            storage_key_type old_key = *const_cast<storage_key_type *>(key_dst);
            if constexpr (compare_key && key_is_vec) {
              for (typename original_key_type::index_type i = 0; i != original_key_type::extent;
                   ++i)
                key_dst->data()[i] = insertion_key.val(i);
            } else
              *key_dst = insertion_key;

            if (insertion_index == sentinel_v) {
              no = atomic_add(omp_c, _cnt, (size_type)1);
              insertion_index = no;
            }
            auto old_index = atomic_exch(omp_c, &_table.indices[bucket_offset + random_location],
                                         insertion_index);

            thread_fence(omp_c);
            atomic_exch(omp_c, &_table.status[bucket_offset / bucket_size], -1);
            // should be old keys instead, not (h)ashed keys
            auto bucket0 = reinterpret_bits<mapped_hashed_key_type>(_hf0(old_key)) % _numBuckets;
            auto bucket1 = reinterpret_bits<mapped_hashed_key_type>(_hf1(old_key)) % _numBuckets;
            auto bucket2 = reinterpret_bits<mapped_hashed_key_type>(_hf2(old_key)) % _numBuckets;

            auto new_bucket_id = bucket0;
            new_bucket_id = bucket_offset == bucket1 * bucket_size ? bucket2 : new_bucket_id;
            new_bucket_id = bucket_offset == bucket0 * bucket_size ? bucket1 : new_bucket_id;

            bucket_offset = new_bucket_id * bucket_size;

            insertion_key = old_key;
            insertion_index = old_index;
          }
          cuckoo_counter++;
        }
      } while (cuckoo_counter < _maxCuckooChains);
      return failure_token_v;
    }

    template <bool retrieve_index = true, execspace_e S = space,
              enable_if_all<S == execspace_e::openmp> = 0>
    [[nodiscard]] inline index_type query(const original_key_type &key,
                                          wrapv<retrieve_index> = {}) const noexcept {
      if (_numBuckets == 0) {
        if constexpr (retrieve_index)
          return sentinel_v;
        else
          return detail::deduce_numeric_max<index_type>();
      }
      auto bucket_offset
          = reinterpret_bits<mapped_hashed_key_type>(_hf0(key)) % _numBuckets * bucket_size;
      storage_key_type query_key = transKey(key);
      for (int iter = 0; iter < 3; ++iter) {
        int location = -1;
        for (int i = 0; i != bucket_size; ++i)
          if (equal_to{}(query_key, _table.keys[bucket_offset + i])) {
            location = i;
            break;
          }
        if (location != -1) {
          if constexpr (retrieve_index) {
            index_type found_value = _table.indices[bucket_offset + location];
            return found_value;
          } else
            return bucket_offset + location;
        } else {
          int load = 0;
          for (int i = 0; i != bucket_size; ++i)
            if (!equal_to{}(compare_key_sentinel_v, _table.keys[bucket_offset + i])) ++load;

          if (load < bucket_size)
            return sentinel_v;
          else
            bucket_offset = iter == 0 ? reinterpret_bits<mapped_hashed_key_type>(_hf1(key))
                                            % _numBuckets * bucket_size
                                      : reinterpret_bits<mapped_hashed_key_type>(_hf2(key))
                                            % _numBuckets * bucket_size;
        }
      }
      if constexpr (retrieve_index)
        return sentinel_v;
      else
        return detail::deduce_numeric_max<index_type>();
    }

    ///
    /// entry (return the location of the key)
    ///
    template <execspace_e S = space, enable_if_all<S == execspace_e::openmp> = 0>
    [[nodiscard]] inline index_type entry(const original_key_type &find_key) const noexcept {
      return query(find_key, false_c);
    }
#endif

    table_t _table;
    conditional_t<is_const_structure,
                  VectorView<space, const Vector<key_type, allocator_type>, Base>,
                  VectorView<space, Vector<key_type, allocator_type>, Base>>
        _activeKeys;
    // conditional_t<is_const_structure, const key_type *, key_type *> _activeKeys;
    conditional_t<is_const_structure, const size_type *, size_type *> _cnt;
    conditional_t<is_const_structure, const u8 *, u8 *> _success;
    size_type _numBuckets;
    u32 _maxCuckooChains;
    hasher_type _hf0, _hf1, _hf2;
  };

  template <execspace_e ExecSpace, typename KeyT, typename Index, bool KeyCompare, typename Hasher,
            int B, typename AllocatorT, bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(bcht<KeyT, Index, KeyCompare, Hasher, B, AllocatorT> &table,
                      wrapv<Base> = {}) {
    return BCHTView<ExecSpace, bcht<KeyT, Index, KeyCompare, Hasher, B, AllocatorT>, Base>{table};
  }
  template <execspace_e ExecSpace, typename KeyT, typename Index, bool KeyCompare, typename Hasher,
            int B, typename AllocatorT, bool Base = !ZS_ENABLE_OFB_ACCESS_CHECK>
  decltype(auto) view(const bcht<KeyT, Index, KeyCompare, Hasher, B, AllocatorT> &table,
                      wrapv<Base> = {}) {
    return BCHTView<ExecSpace, const bcht<KeyT, Index, KeyCompare, Hasher, B, AllocatorT>, Base>{
        table};
  }

  template <execspace_e space, typename KeyT, typename Index, bool KeyCompare, typename Hasher,
            int B, typename AllocatorT>
  decltype(auto) proxy(bcht<KeyT, Index, KeyCompare, Hasher, B, AllocatorT> &table) {
    return view<space>(table, false_c);
  }
  template <execspace_e space, typename KeyT, typename Index, bool KeyCompare, typename Hasher,
            int B, typename AllocatorT>
  decltype(auto) proxy(const bcht<KeyT, Index, KeyCompare, Hasher, B, AllocatorT> &table) {
    return view<space>(table, false_c);
  }

}  // namespace zs