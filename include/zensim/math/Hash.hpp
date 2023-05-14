#pragma once
#include "zensim/meta/Meta.h"

namespace zs {

  /*
    //  on why XOR is not a good choice for hash-combining:
    //  https://stackoverflow.com/questions/5889238/why-is-xor-the-default-way-to-combine-hashes
    //
    //  this is from boost
    //
    template <typename T> constexpr void hash_combine(size_t &seed, const T &val) {
      seed ^= (val + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    }
    */
  // ref: https://github.com/HowardHinnant/hash_append/issues/7
  template <typename Tn, typename T,
            enable_if_all<is_unsigned_v<Tn>, is_integral_v<T>> = 0>
  constexpr void hash_combine(Tn &seed, const T &val) {
    static_assert(sizeof(Tn) >= 2 && sizeof(Tn) <= 8, "Tn should contain at least 16 bits");
    if constexpr (sizeof(Tn) == 2)
      seed ^= (val + 0x9e37U + (seed << 3) + (seed >> 1));
    else if constexpr (sizeof(Tn) == 4)
      seed ^= (val + 0x9e3779b9U + (seed << 6) + (seed >> 2));
    else if constexpr (sizeof(Tn) == 8)
      seed ^= (val + 0x9e3779b97f4a7c15LLU + (seed << 12) + (seed >> 4));
  }

/// https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
#if 0
constexpr uint32_t hash(uint32_t x) noexcept {
  x += (x << 10u);
  x ^= (x >> 6u);
  x += (x << 3u);
  x ^= (x >> 11u);
  x += (x << 15u);
  return x;
}
#endif
  template <typename T> constexpr T hash(T x) noexcept {
    if constexpr (sizeof(T) == 4) {
      x = ((x >> 16) ^ x) * 0x45d9f3b;
      x = ((x >> 16) ^ x) * 0x45d9f3b;
      x = (x >> 16) ^ x;
    } else if constexpr (sizeof(T) == 8) {
      x = (x ^ (x >> 30)) * uint64_t(0xbf58476d1ce4e5b9);
      x = (x ^ (x >> 27)) * uint64_t(0x94d049bb133111eb);
      x = x ^ (x >> 31);
    }
    return x;
  }
  template <typename T> constexpr T unhash(T x) noexcept {
    if constexpr (sizeof(T) == 4) {
      x = ((x >> 16) ^ x) * 0x119de1f3;
      x = ((x >> 16) ^ x) * 0x119de1f3;
      x = (x >> 16) ^ x;
    } else if constexpr (sizeof(T) == 8) {
      x = (x ^ (x >> 31) ^ (x >> 62)) * uint64_t(0x319642b2d24d8ec3);
      x = (x ^ (x >> 27) ^ (x >> 54)) * uint64_t(0x96de1b173f119089);
      x = x ^ (x >> 30) ^ (x >> 60);
    }
    return x;
  }

}  // namespace zs
