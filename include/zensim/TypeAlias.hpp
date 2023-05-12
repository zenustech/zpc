#pragma once
#include "Platform.hpp"
#include "ZpcMeta.hpp"

namespace zs {

#define FLT_RADIX 2
#define FLT_MANT_DIG 24
#define DBL_MANT_DIG 53
#define FLT_DIG 6
#define DBL_DIG 15
#define FLT_MIN_EXP -125
#define DBL_MIN_EXP -1021
#define FLT_MIN_10_EXP -37
#define DBL_MIN_10_EXP -307
#define FLT_MAX_EXP 128
#define DBL_MAX_EXP 1024
#define FLT_MAX_10_EXP 38
#define DBL_MAX_10_EXP 308
#define FLT_MAX 3.4028234e38f
#define DBL_MAX 1.7976931348623157e308
#define FLT_EPSILON 1.19209289e-7f
#define DBL_EPSILON 2.220440492503130e-16
#define FLT_MIN 1.1754943e-38f
#define DBL_MIN 2.2250738585072013e-308
#define FLT_ROUNDS 1

  using uint = unsigned int;
  // signed
  using i8 = signed char;
  using i16 = signed short;
  using i32 = signed int;
  using i64 = signed long long int;
  static_assert(sizeof(i8) == 1 && sizeof(i16) == 2 && sizeof(i32) == 4 && sizeof(i64) == 8,
                "these signed integers are not of the sizes expected!");
  // unsigned
  using u8 = unsigned char;
  using u16 = unsigned short;
  using u32 = unsigned int;
  using u64 = unsigned long long int;
  static_assert(sizeof(u8) == 1 && sizeof(u16) == 2 && sizeof(u32) == 4 && sizeof(u64) == 8,
                "these unsigned integers are not of the sizes expected!");
  // floating points
  using f32 = float;
  using f64 = double;
  static_assert(sizeof(f32) == 4 && sizeof(f64) == 8,
                "these floating points are not of the sizes expected!");

  union dat32 {
    f32 f;
    i32 i;
    u32 u;
    template <typename T> constexpr T &cast() noexcept;
    template <typename T> constexpr T cast() const noexcept;
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
    template <typename T> constexpr T &cast() noexcept;
    template <typename T> constexpr T cast() const noexcept;
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
  template <typename T> using RefPtr = decay_t<T> *;             ///< non-owning reference
  template <typename T> using ConstRefPtr = const decay_t<T> *;  ///< non-owning const reference
  // template <typename T> using Holder = ::std::unique_ptr<T>;
  // template <typename T> using SharedHolder = ::std::shared_ptr<T>;

  using NodeID = i32;
  /// @note processor index in the residing node
  using ProcID = i8;
  /// @note negative stream identified as the default stream
  using StreamID = i32;
  using EventID = u32;

  constexpr void do_nothing(...) noexcept {}
  struct do_nothing_op {
    constexpr void operator()(...) noexcept {}
  };

}  // namespace zs

/// lambda capture

#if ZS_ENABLE_CUDA && defined(__CUDACC__)
#  if defined(ZS_LAMBDA)
#    undef ZS_LAMBDA
#  endif
#  define ZS_LAMBDA __device__
#else
#  if defined(ZS_LAMBDA)
#    undef ZS_LAMBDA
#  endif
#  define ZS_LAMBDA
#endif

#if ZS_ENABLE_CUDA && defined(__CUDACC__)
#  if defined(ZS_FUNCTION)
#    undef ZS_FUNCTION
#  endif
#  define ZS_FUNCTION __forceinline__ __device__
#else
#  if defined(ZS_FUNCTION)
#    undef ZS_FUNCTION
#  endif
#  define ZS_FUNCTION inline
#endif