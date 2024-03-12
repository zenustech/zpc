#pragma once
#include "zensim/ZpcMeta.hpp"
#include "zensim/math/VecInterface.hpp"

namespace zs {

  template <typename KeyT> struct universal_hash_base {
    using key_type = KeyT;
    using result_type = u32;

    // borrowed from cudpp
    static constexpr u32 prime_divisor = 4294967291u;

    universal_hash_base() noexcept = default;
    constexpr universal_hash_base(u32 hash_x, u32 hash_y) : _hashx(hash_x), _hashy(hash_y) {}
    ~universal_hash_base() = default;
    universal_hash_base(const universal_hash_base &) = default;
    universal_hash_base(universal_hash_base &&) = default;
    universal_hash_base &operator=(const universal_hash_base &) = default;
    universal_hash_base &operator=(universal_hash_base &&) = default;

    template <bool isVec = is_vec<key_type>::value, enable_if_t<!isVec> = 0>
    constexpr result_type operator()(const key_type &key) const noexcept {
      return (((_hashx ^ key) + _hashy) % prime_divisor);
    }
    template <typename VecT, enable_if_t<is_integral_v<typename VecT::value_type>> = 0>
    constexpr result_type operator()(const VecInterface<VecT> &key) const noexcept {
      static_assert(VecT::extent >= 1, "should at least have one element");
      const universal_hash_base<typename VecT::value_type> subhasher{_hashx, _hashy};
      u32 ret = subhasher(key.val(0));
      for (typename VecT::index_type d = 1; d != VecT::extent; ++d)
        hash_combine(ret, subhasher(key.val(d)));
      return ret;
    }
    template <bool isVec = is_vec<key_type>::value, enable_if_t<isVec> = 0>
    constexpr result_type operator()(const key_type &key) const noexcept {
      static_assert(key_type::extent >= 1, "should at least have one element");
      const universal_hash_base<typename key_type::value_type> subhasher{_hashx, _hashy};
      u32 ret = subhasher(key[0]);
      for (typename key_type::index_type d = 1; d != key_type::extent; ++d)
        hash_combine(ret, subhasher(key.val(d)));
      return ret;
    }

    u32 _hashx;
    u32 _hashy;
  };

  template <typename key_type> struct alignas(next_2pow(sizeof(key_type))) storage_key_type_impl {
    static constexpr size_t num_total_bytes = next_2pow(sizeof(key_type));
    static constexpr size_t num_padded_bytes = next_2pow(sizeof(key_type)) - sizeof(key_type);
    constexpr storage_key_type_impl() noexcept = default;
    constexpr storage_key_type_impl(const key_type &k) noexcept : val{k} {}
    constexpr storage_key_type_impl(key_type &&k) noexcept : val{move(k)} {}
    ~storage_key_type_impl() = default;
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

#if ZS_ENABLE_SERIALIZATION
    template <typename S> void serialize(S &s) { serialize(s, val); }
#endif
  };

}  // namespace zs