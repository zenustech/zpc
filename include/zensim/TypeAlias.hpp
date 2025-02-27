#pragma once
#include "zensim/Platform.hpp"
#include "zensim/ZpcMeta.hpp"

namespace zs {

#define ZS_FLT_RADIX 2
#define ZS_FLT_MANT_DIG 24
#define ZS_DBL_MANT_DIG 53
#define ZS_FLT_DIG 6
#define ZS_DBL_DIG 15
#define ZS_FLT_MIN_EXP -125
#define ZS_DBL_MIN_EXP -1021
#define ZS_FLT_MIN_10_EXP -37
#define ZS_DBL_MIN_10_EXP -307
#define ZS_FLT_MAX_EXP 128
#define ZS_DBL_MAX_EXP 1024
#define ZS_FLT_MAX_10_EXP 38
#define ZS_DBL_MAX_10_EXP 308
#define ZS_FLT_MAX 3.4028234e38f
#define ZS_DBL_MAX 1.7976931348623157e308
#define ZS_FLT_EPSILON 1.19209289e-7f
#define ZS_DBL_EPSILON 2.220440492503130e-16
#define ZS_FLT_MIN 1.1754943e-38f
#define ZS_DBL_MIN 2.2250738585072013e-308
#define ZS_FLT_ROUNDS 1

  /// arithmetic type
  constexpr wrapt<u8> u8_c{};
  constexpr wrapt<int> int_c{};
  constexpr wrapt<uint> uint_c{};
  constexpr wrapt<i16> i16_c{};
  constexpr wrapt<i32> i32_c{};
  constexpr wrapt<i64> i64_c{};
  constexpr wrapt<u16> u16_c{};
  constexpr wrapt<u32> u32_c{};
  constexpr wrapt<u64> u64_c{};
  constexpr wrapt<f32> f32_c{};
  constexpr wrapt<float> float_c{};
  constexpr wrapt<f64> f64_c{};
  constexpr wrapt<double> double_c{};
  template <typename T> constexpr wrapt<enable_if_type<is_arithmetic_v<T>, T>> number_c{};

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
#elif ZS_ENABLE_MUSA && defined(__MUSACC__)
#  if defined(ZS_LAMBDA)
#    undef ZS_LAMBDA
#  endif
#  define ZS_LAMBDA __device__
#elif ZS_ENABLE_ROCM && defined(__HIPCC__)
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
#elif ZS_ENABLE_MUSA && defined(__MUSACC__)
#  if defined(ZS_FUNCTION)
#    undef ZS_FUNCTION
#  endif
#  define ZS_FUNCTION __forceinline__ __device__
#elif ZS_ENABLE_ROCM && defined(__HIPCC__)
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