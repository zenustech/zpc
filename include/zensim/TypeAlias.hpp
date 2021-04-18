#pragma once
#include <cstdint>
#include <memory>

// #include "zensim/meta/ControlFlow.h"

namespace zs {

  using uint = unsigned int;
  // using i16 = conditional_t<sizeof(short) == 2, short, int16_t>;
  using i16 = int16_t;
  using i32 = int32_t;
  using i64 = int64_t;
  using u16 = uint16_t;
  using u32 = uint32_t;
  using u64 = uint64_t;
  using f32 = float;
  using f64 = double;

  union dat32 {
    f32 f;
    i32 i;
    u32 u;
    template <typename T> T &cast() noexcept;
    template <typename T> T cast() const noexcept;
    constexpr f32 &asFloat() noexcept { return f; }
    constexpr i32 &asSignedInteger() noexcept { return i; }
    constexpr u32 &asUnsignedInteger() noexcept { return u; }
    constexpr f32 asFloat() const noexcept { return f; }
    constexpr i32 asSignedInteger() const noexcept { return i; }
    constexpr u32 asUnsignedInteger() const noexcept { return u; }
  };
  template <> constexpr f32 &dat32::cast<f32>() noexcept { return f; }
  template <> constexpr i32 &dat32::cast<i32>() noexcept { return i; }
  template <> constexpr u32 &dat32::cast<u32>() noexcept { return u; }
  template <> constexpr f32 dat32::cast<f32>() const noexcept { return f; }
  template <> constexpr i32 dat32::cast<i32>() const noexcept { return i; }
  template <> constexpr u32 dat32::cast<u32>() const noexcept { return u; }

  union dat64 {
    f64 d;
    i64 l;
    u64 ul;
    template <typename T> T &cast() noexcept;
    template <typename T> T cast() const noexcept;
    constexpr f64 &asFloat() noexcept { return d; }
    constexpr i64 &asSignedInteger() noexcept { return l; }
    constexpr u64 &asUnsignedInteger() noexcept { return ul; }
    constexpr f64 asFloat() const noexcept { return d; }
    constexpr i64 asSignedInteger() const noexcept { return l; }
    constexpr u64 asUnsignedInteger() const noexcept { return ul; }
  };
  template <> constexpr f64 &dat64::cast<f64>() noexcept { return d; }
  template <> constexpr i64 &dat64::cast<i64>() noexcept { return l; }
  template <> constexpr u64 &dat64::cast<u64>() noexcept { return ul; }
  template <> constexpr f64 dat64::cast<f64>() const noexcept { return d; }
  template <> constexpr i64 dat64::cast<i64>() const noexcept { return l; }
  template <> constexpr u64 dat64::cast<u64>() const noexcept { return ul; }

  // kokkos::ObservingRawPtr<T>, OptionalRef<T>
  // vsg::ref_ptr<T>
  template <typename T> using RefPtr = std::decay_t<T> *;             ///< non-owning reference
  template <typename T> using ConstRefPtr = const std::decay_t<T> *;  ///< non-owning reference
  template <typename T> using Holder = std::unique_ptr<T>;            ///< non-owning reference

  using NodeID = i32;
  using ProcID = char;
  using StreamID = u32;
  using EventID = u32;

/// lambda capture
/// https://vittorioromeo.info/index/blog/capturing_perfectly_forwarded_objects_in_lambdas.html
#define FWD(...) ::std::forward<decltype(__VA_ARGS__)>(__VA_ARGS__)

}  // namespace zs
