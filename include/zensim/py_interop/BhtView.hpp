#pragma once
#include "zensim/ZpcMeta.hpp"
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
    // static constexpr index_type key_scalar_sentinel_v = limits<index_type>::max(); //
    // detail::deduce_numeric_max<index_type>();
    static constexpr value_type sentinel_v{-1};  // this requires key_type to be signed type
    static constexpr status_type status_sentinel_v{-1};
    static constexpr value_type failure_token_v = detail::deduce_numeric_lowest<value_type>();

    BhtViewLite() noexcept = default;
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