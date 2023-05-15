#pragma once
#include "ZpcFunctional.hpp"
#include "types/Property.h"

#if defined(__CUDACC__)

#else
extern "C" {
// stdio.h
int printf(const char *format, ...);

// stdlib.h
int abs(int);
long long llabs(long long);

// math.h
float copysignf(float x, float y);
double copysign(double x, double y);
float fabsf(float);
double fabs(double);
float fmaxf(float x, float y);
double fmax(double x, double y);
float fminf(float x, float y);
double fmin(double x, double y);
float fmaf(float, float, float);
double fma(double x, double y, double z);
float fmodf(float, float);
double fmod(double, double);
float ceilf(float);
double ceil(double);
float floorf(float);
double floor(double);
float sqrtf(float);
double sqrt(double);
float logf(float);
double log(double);
float log1pf(float);
double log1p(double);
float expf(float);
double exp(double);
float powf(float, float);
double pow(double, double);

float sinhf(float);
double sinh(double);
float sinf(float);
double sin(double);
float asinhf(float);
double asinh(double);
float asinf(float);
double asin(double);

float cosf(float);
double cos(double);
float coshf(float);
double cosh(double);
float acoshf(float);
double acosh(double);
float acosf(float);
double acos(double);

float atan2f(float, float);
double atan2(double, double);

float modff(float arg, float *iptr);
double modf(double arg, double *iptr);
float frexpf(float arg, int *exp);
double frexp(double arg, int *exp);
float ldexpf(float arg, int exp);
double ldexp(double arg, int exp);

// not yet wrapped
float log2f(float);
double log2(double);
float log10f(float);
double log10(double);
float roundf(float);
double round(double);
float truncf(float);
double trunc(double);
float atanf(float);
double atan(double);
float tanf(float);
double tan(double);
float tanhf(float);
double tanh(double);
// float rintf(float);
// double rint(double);
}
#endif

namespace zs {

  namespace mathutil_impl {
    // constexpr scan only available in c++20:
    // https://en.cppreference.com/w/cpp/algorithm/exclusive_scan
    template <typename... Args, auto... Is>
    constexpr auto incl_prefix_sum_impl(sint_t I, index_sequence<Is...>, Args &&...args) noexcept {
      return (((sint_t)Is <= I ? forward<Args>(args) : 0) + ...);
    }
    template <typename... Args, auto... Is>
    constexpr auto excl_prefix_sum_impl(sint_t I, index_sequence<Is...>, Args &&...args) noexcept {
      return (((sint_t)Is < I ? forward<Args>(args) : 0) + ...);
    }
    template <typename... Args, auto... Is>
    constexpr auto excl_suffix_mul_impl(sint_t I, index_sequence<Is...>, Args &&...args) noexcept {
      return (((sint_t)Is > I ? forward<Args>(args) : 1) * ...);
    }
  }  // namespace mathutil_impl

  /// copied from gcem_options.hpp
  constexpr double g_pi = 3.1415926535897932384626433832795028841972L;
  constexpr double g_half_pi = 1.5707963267948966192313216916397514420986L;
  constexpr double g_sqrt2 = 1.4142135623730950488016887242096980785697L;

  namespace math {

    template <typename T, enable_if_t<is_floating_point_v<T>> = 0>
    constexpr bool near_zero(T v) noexcept {
      constexpr auto eps = (T)128 * zs::detail::deduce_numeric_epsilon<T>();
      return v >= -eps && v <= eps;
    }

    template <typename T, enable_if_t<is_fundamental_v<T>> = 0> constexpr T min(T x, T y) noexcept {
      return y < x ? y : x;
    }
    template <typename T, enable_if_t<is_fundamental_v<T>> = 0> constexpr T max(T x, T y) noexcept {
      return y > x ? y : x;
    }
    template <typename T, enable_if_t<is_fundamental_v<T>> = 0> constexpr T abs(T x) noexcept {
      return x < 0 ? -x : x;
    }
    // TODO refer to:
    // https://github.com/mountunion/ModGCD-OneGPU/blob/master/ModGCD-OneGPU.pdf
    // http://www.iaeng.org/IJCS/issues_v42/issue_4/IJCS_42_4_01.pdf
    template <typename Ti, enable_if_t<is_integral_v<Ti>> = 0>
    constexpr Ti gcd(Ti u, Ti v) noexcept {
      while (v != 0) {
        auto r = u % v;
        u = v;
        v = r;
      }
      return u;
    }
    template <typename Ti, enable_if_t<is_integral_v<Ti>> = 0>
    constexpr Ti lcm(Ti u, Ti v) noexcept {
      return (u / gcd(u, v)) * v;
    }

    /// binary_op_result
    template <typename T0, typename T1> struct binary_op_result {
      template <typename A = T0, typename B = T1,
                enable_if_all<is_integral_v<A>, is_integral_v<B>> = 0>
      static auto determine_type() -> conditional_t<
          is_signed_v<A> && is_signed_v<B>, conditional_t<(sizeof(A) >= sizeof(B)), A, B>,
          conditional_t<
              is_signed_v<A>, A,
              conditional_t<is_signed_v<B>, B, conditional_t<(sizeof(A) >= sizeof(B)), A, B>>>>;
      template <typename A = T0, typename B = T1,
                enable_if_t<!is_integral_v<A> || !is_integral_v<B>> = 0>
      static auto determine_type() -> common_type_t<A, B>;
      using type = decltype(determine_type());
    };
    template <typename T0, typename T1> using binary_op_result_t =
        typename binary_op_result<T0, T1>::type;

    /// op_result
    template <typename... Ts> struct op_result;
    template <typename T> struct op_result<T> {
      using type = T;
    };
    template <typename T, typename... Ts> struct op_result<T, Ts...> {
      using type = binary_op_result_t<T, typename op_result<Ts...>::type>;
    };
    /// @brief determine the most appropriate resulting type of a binary operation
    template <typename... Args> using op_result_t = typename op_result<Args...>::type;
  }  // namespace math

  /**
   *  math intrinsics (not constexpr at all! just cheating the compiler)
   */
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T copysign(T mag, T sgn, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::copysignf(mag, sgn);
      else
        return ::copysign((double)mag, (double)sgn);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [copysign] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::copysignf(mag, sgn);
      else
        return ::copysign((double)mag, (double)sgn);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T abs(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::fabsf(v);
      else
        return ::fabs((double)v);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [abs] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::fabsf(v);
      else
        return ::fabs((double)v);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T max(T x, T y, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::fmaxf(x, y);
      else
        return ::fmax((double)x, (double)y);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [max] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::fmaxf(x, y);
      else
        return ::fmax((double)x, (double)y);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T min(T x, T y, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::fminf(x, y);
      else
        return ::fmin((double)x, (double)y);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [min] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::fminf(x, y);
      else
        return ::fmin((double)x, (double)y);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T fma(T x, T y, T z, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::fmaf(x, y, z);
      else
        return ::fma((double)x, (double)y, (double)z);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [fma] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::fmaf(x, y, z);
      else
        return ::fma((double)x, (double)y, (double)z);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T fmod(T x, T y, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::fmodf(x, y);
      else
        return ::fmod((double)x, (double)y);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [fmod] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::fmodf(x, y);
      else
        return ::fmod((double)x, (double)y);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T ceil(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::ceilf(v);
      else
        return ::ceil((double)v);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [ceil] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::ceilf(v);
      else
        return ::ceil((double)v);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T floor(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::floorf(v);
      else
        return ::floor((double)v);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [floor] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::floorf(v);
      else
        return ::floor((double)v);
#endif
    }
  }

  // different from math::sqrt
  template <typename T, enable_if_t<is_arithmetic_v<T>> = 0> constexpr T sqr(T v) noexcept {
    return v * v;
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T sqrt(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::sqrtf(v);
      else
        return ::sqrt((double)v);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [sqrt] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::sqrtf(v);
      else
        return ::sqrt((double)v);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T rsqrt(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::rsqrtf(v);
      else
        return ::rsqrt((double)v);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [rsqrt] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return (T)1 / (T)::sqrtf(v);
      else
        return (T)1 / (T)::sqrt((double)v);
#endif
    }
  }

  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T log(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::logf(v);
      else
        return ::log((double)v);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [log] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::logf(v);
      else
        return ::log((double)v);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T log1p(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::log1pf(v);
      else
        return ::log1p((double)v);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [log1p] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::log1pf(v);
      else
        return ::log1p((double)v);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T exp(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::expf(v);
      else
        return ::exp((double)v);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [exp] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::expf(v);
      else
        return ::exp((double)v);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T pow(T base, T exp, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::powf(base, exp);
      else
        return ::pow((double)base, (double)exp);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [pow] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::powf(base, exp);
      else
        return ::pow((double)base, (double)exp);
#endif
    }
  }

  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  ZS_FUNCTION T add_ru(T x, T y, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::__fadd_ru(x, y);
      else
        return ::__dadd_ru((double)x, (double)y);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [add_ru] is missing!");
      return 0;
#endif
    } else
      /// @note refer to https://en.cppreference.com/w/cpp/numeric/fenv/FE_round
      return (x + y);
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  ZS_FUNCTION T sub_ru(T x, T y, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::__fsub_ru(x, y);
      else
        return ::__dsub_ru((double)x, (double)y);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [sub_ru] is missing!");
      return 0;
#endif
    } else
      /// @note refer to https://en.cppreference.com/w/cpp/numeric/fenv/FE_round
      return (x - y);
  }

  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T sinh(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::sinhf(v);
      else
        return ::sinh((double)v);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [sinh] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::sinhf(v);
      else
        return ::sinh((double)v);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T sin(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::sinf(v);
      else
        return ::sin((double)v);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [sin] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::sinf(v);
      else
        return ::sin((double)v);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T asinh(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::asinhf(v);
      else
        return ::asinh((double)v);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [asinh] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::asinhf(v);
      else
        return ::asinh((double)v);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T asin(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::asinf(v);
      else
        return ::asin((double)v);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [asin] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::asinf(v);
      else
        return ::asin((double)v);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T cosh(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::coshf(v);
      else
        return ::cosh((double)v);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [cosh] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::coshf(v);
      else
        return ::cosh((double)v);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T cos(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::cosf(v);
      else
        return ::cos((double)v);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [cos] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::cosf(v);
      else
        return ::cos((double)v);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T acosh(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::acoshf(v);
      else
        return ::acosh((double)v);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [acosh] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::acoshf(v);
      else
        return ::acosh((double)v);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T acos(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::acosf(v);
      else
        return ::acos((double)v);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [acos] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::acosf(v);
      else
        return ::acos((double)v);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T atan2(T y, T x, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::atan2f(y, x);
      else
        return ::atan2((double)y, (double)x);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [atan2] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::atan2f(y, x);
      else
        return ::atan2((double)y, (double)x);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr bool isnan(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      return ::isnan(v) != 0;  // due to msvc
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [isnan] is missing!");
      return 0;
#endif
    } else
      // https://en.cppreference.com/w/c/numeric/math/isnan
      return v != v;
  }
#if 0
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr bool isinf(T v, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#  if defined(__CUDACC__)
      return ::isinf(v) != 0;  // due to msvc
#  else
      static_assert(space != execspace_e::cuda, "cuda implementation of [isinf] is missing!");
      return 0;
#  endif
    } else
      return std::isinf(v);
  }
#endif

  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T modf(T x, T *iptr, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      static_assert(is_same_v<T, float> || is_same_v<T, double>, "modf only supports float/double");
      if constexpr (is_same_v<T, float>)
        return ::modff(x, iptr);
      else if constexpr (is_same_v<T, double>)
        return ::modf(x, iptr);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [modf] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      static_assert(is_same_v<T, float> || is_same_v<T, double>, "modf only supports float/double");
      if constexpr (is_same_v<T, float>)
        return ::modff(x, iptr);
      else if constexpr (is_same_v<T, double>)
        return ::modf(x, iptr);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T frexp(T x, int *exp, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::frexpf(x, exp);
      else
        return ::frexp((double)x, exp);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [frexp] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::frexpf(x, exp);
      else
        return ::frexp((double)x, exp);
#endif
    }
  }
  template <typename T, execspace_e space = deduce_execution_space(),
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr T ldexp(T x, int exp, wrapv<space> = {}) noexcept {
    if constexpr (space == execspace_e::cuda) {
#if defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::ldexpf(x, exp);  // scalbnf(x, exp)
      else
        return ::ldexp((double)x, exp);
#else
      static_assert(space != execspace_e::cuda, "cuda implementation of [ldexp] is missing!");
      return 0;
#endif
    } else {
#if !defined(__CUDACC__)
      if constexpr (is_same_v<T, float>)
        return ::ldexpf(x, exp);  // scalbnf(x, exp)
      else
        return ::ldexp((double)x, exp);
#endif
    }
  }

  ///
  template <typename T, typename Data, enable_if_t<is_floating_point_v<T>> = 0>
  constexpr auto linear_interop(T alpha, Data &&a, Data &&b) noexcept {
    return a + (b - a) * alpha;
  }

}  // namespace zs