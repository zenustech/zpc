#pragma once

#include <cmath>
#include <limits>

#include "zensim/ZpcFunctional.hpp"
#include "zensim/ZpcMathUtils.hpp"
#include "zensim/types/Property.h"
#if defined(__CUDACC__) || defined(__MUSACC__) || defined(__HIPCC__)
#  include "math.h"  // CUDA/MUSA/ROCm math library
#endif
#include <functional>
#include <type_traits>
#include <utility>

#include "Complex.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/meta/Meta.h"

namespace zs {

  template <typename T> using limits = std::numeric_limits<T>;

  // 26.2.7/3 abs(__z):  Returns the magnitude of __z.
  template <typename T> constexpr T abs(const complex<T> &z) noexcept {
    T x = z.real();
    T y = z.imag();
    const T s = zs::max(zs::abs(x), zs::abs(y));
    if (s == T{}) return s;
    x /= s;
    y /= s;
    return s * zs::sqrt(x * x + y * y);
  }
  // 26.2.7/4: arg(__z): Returns the phase angle of __z.
  template <typename T> constexpr T arg(const complex<T> &z) noexcept {
    return zs::atan2(z.imag(), z.real());
  }
  template <typename T> constexpr complex<T> polar(const T &rho, const T &theta) {
    // assert(rho >= 0);
    return complex<T>{rho * zs::cos(theta), rho * zs::sin(theta)};
  }

  // 26.2.8/1 cos(__z):  Returns the cosine of __z.
  template <typename T> constexpr complex<T> cos(const complex<T> &z) {
    const T x = z.real();
    const T y = z.imag();
    return complex<T>{zs::cos(x) * zs::cosh(y), -zs::sin(x) * zs::sinh(y)};
  }
  template <typename T> constexpr complex<T> cosh(const complex<T> &z) {
    const T x = z.real();
    const T y = z.imag();
    return complex<T>{zs::cosh(x) * zs::cos(y), zs::sinh(x) * zs::sin(y)};
  }

  // 26.2.8/3 exp(__z): Returns the complex base e exponential of x
  template <typename T> constexpr complex<T> exp(const complex<T> &z) {
    return zs::polar(zs::exp(z.real()), z.imag());
  }

  // 26.2.8/5 log(__z): Returns the natural complex logarithm of __z.
  //                    The branch cut is along the negative axis.
  template <typename T> constexpr complex<T> log(const complex<T> &z) {
    return complex<T>{zs::log(zs::abs(z)), zs::arg(z)};
  }
  template <typename T> constexpr complex<T> log10(const complex<T> &z) {
    return zs::log(z) / zs::log((T)10.0);
  }

  // 26.2.8/10 sin(__z): Returns the sine of __z.
  template <typename T> constexpr complex<T> sin(const complex<T> &z) {
    const T x = z.real();
    const T y = z.imag();
    return complex<T>{zs::sin(x) * zs::cosh(y), zs::cos(x) * zs::sinh(y)};
  }

  // 26.2.8/11 sinh(__z): Returns the hyperbolic sine of __z.
  template <typename T> constexpr complex<T> sinh(const complex<T> &z) {
    const T x = z.real();
    const T y = z.imag();
    return complex<T>{zs::sinh(x) * zs::cos(y), zs::cosh(x) * zs::sin(y)};
  }

  // 26.2.8/13 sqrt(__z): Returns the complex square root of __z.
  //                     The branch cut is on the negative axis.
  template <typename T> constexpr complex<T> sqrt(const complex<T> &z) {
    T x = z.real();
    T y = z.imag();
    if (x == T{}) {
      T t = zs::sqrt(zs::abs(y) / 2);
      return complex<T>{t, y < T{} ? -t : t};
    } else {
      T t = zs::sqrt(2 * (zs::abs(z) + zs::abs(x)));
      T u = t / 2;
      return x > T{} ? complex<T>{u, y / t} : complex<T>{zs::abs(y) / t, y < T{} ? -u : u};
    }
  }

  // 26.2.8/14 tan(__z):  Return the complex tangent of __z.
  template <typename T> constexpr complex<T> tan(const complex<T> &z) {
    return zs::sin(z) / zs::cos(z);
  }

  // 26.2.8/15 tanh(__z):  Returns the hyperbolic tangent of __z.
  template <typename T> constexpr complex<T> tanh(const complex<T> &z) {
    return zs::sinh(z) / zs::cosh(z);
  }

  namespace detail {
    // 26.2.8/9  pow(__x, __y): Returns the complex power base of __x
    //                          raised to the __y-th power.  The branch
    //                          cut is on the negative axis.
    template <typename T>
    constexpr complex<T> __complex_pow_unsigned(const complex<T> &x, unsigned n) {
      complex<T> y = n % 2 ? x : complex<T>(1);

      while (n >>= 1) {
        x *= x;
        if (n % 2) y *= x;
      }
      return y;
    }
  }  // namespace detail

  // In C++11 mode we used to implement the resolution of
  // DR 844. complex pow return type is ambiguous.
  // thus the following overload was disabled in that mode.  However, doing
  // that causes all sorts of issues, see, for example:
  //   http://gcc.gnu.org/ml/libstdc++/2013-01/msg00058.html
  // and also PR57974.
  template <typename T> constexpr complex<T> pow(const complex<T> &z, int n) {
    return n < 0 ? complex<T>{1} / detail::__complex_pow_unsigned(z, (unsigned)-n)
                 : detail::__complex_pow_unsigned(z, n);
  }

  template <typename T> constexpr complex<T> pow(const complex<T> &x, const complex<T> &y) {
    return x == T{} ? T{} : zs::exp(y * zs::log(x));
  }

  template <typename T, enable_if_t<!is_integral_v<T>> = 0>
  constexpr complex<T> pow(const complex<T> &x, const T &y) {
    if (x.imag() == T{} && x.real() > T{}) return zs::pow(x.real(), y);

    complex<T> t = zs::log(x);
    return zs::polar<T>(zs::exp(y * t.real()), y * t.imag());
  }

  template <typename T> constexpr complex<T> pow(const T &x, const complex<T> &y) {
    return x > T{} ? zs::polar<T>(zs::pow(x, y.real()), y.imag() * zs::log(x))
                   : zs::pow(complex<T>{x}, y);
  }

  /// acos(__z) [8.1.2].
  //  Effects:  Behaves the same as C99 function cacos, defined
  //            in subclause 7.3.5.1.
  template <typename T> constexpr complex<T> acos(const complex<T> &z) {
    const complex<T> t = zs::asin(z);
    const T __pi_2 = 1.5707963267948966192313216916397514L;
    return complex<T>{__pi_2 - t.real(), -t.imag()};
  }
  /// asin(__z) [8.1.3].
  //  Effects:  Behaves the same as C99 function casin, defined
  //            in subclause 7.3.5.2.
  template <typename T> constexpr complex<T> asin(const complex<T> &z) {
    complex<T> t{-z.imag(), z.real()};
    t = zs::asinh(t);
    return complex<T>{t.imag(), -t.real()};
  }
  /// atan(__z) [8.1.4].
  //  Effects:  Behaves the same as C99 function catan, defined
  //            in subclause 7.3.5.3.
  template <typename T> constexpr complex<T> atan(const complex<T> &z) {
    const T r2 = z.real() * z.real();
    const T x = (T)1.0 - r2 - z.imag() * z.imag();

    T num = z.imag() + (T)1.0;
    T den = z.imag() - (T)1.0;

    num = r2 + num * num;
    den = r2 + den * den;

    return complex<T>{(T)0.5 * zs::atan2((T)2.0 * z.real(), x), (T)0.25 * zs::log(num / den)};
  }
  /// acosh(__z) [8.1.5].
  //  Effects:  Behaves the same as C99 function cacosh, defined
  //            in subclause 7.3.6.1.
  template <typename T> constexpr complex<T> acosh(const complex<T> &z) {
    // Kahan's formula.
    return (T)2.0 * zs::log(zs::sqrt((T)0.5 * (z + (T)1.0)) + zs::sqrt((T)0.5 * (z - (T)1.0)));
  }
  /// asinh(__z) [8.1.6].
  //  Effects:  Behaves the same as C99 function casin, defined
  //            in subclause 7.3.6.2.
  template <typename T> constexpr complex<T> asinh(const complex<T> &z) {
    complex<T> t{(z.real() - z.imag()) * (z.real() + z.imag()) + (T)1.0,
                 (T)2.0 * z.real() * z.imag()};
    t = zs::sqrt(t);
    return zs::log(t + z);
  }
  /// atanh(__z) [8.1.7].
  //  Effects:  Behaves the same as C99 function catanh, defined
  //            in subclause 7.3.6.3.
  template <typename T> constexpr complex<T> atanh(const complex<T> &z) {
    const T i2 = z.imag() * z.imag();
    const T x = T(1.0) - i2 - z.real() * z.real();

    T num = T(1.0) + z.real();
    T den = T(1.0) - z.real();

    num = i2 + num * num;
    den = i2 + den * den;

    return complex<T>{(T)0.25 * (zs::log(num) - zs::log(den)),
                      (T)0.5 * zs::atan2((T)2.0 * z.imag(), x)};
  }
#if 0
  /// fabs(__z) [8.1.8].
  //  Effects:  Behaves the same as C99 function cabs, defined
  //            in subclause 7.3.8.1.
  template <typename T> constexpr T fabs(const complex<T> &z) { return zs::abs(z); }
#endif

  /// additional overloads [8.1.9]
  template <typename T> constexpr auto arg(T x) {
    static_assert(is_floating_point_v<T> || is_integral_v<T>, "invalid param type for func [arg]");
    using type = conditional_t<is_floating_point_v<T>, T, double>;
    return zs::arg(complex<type>{x});
  }
  // ignore the remaining type promotions for now

  /// customized zpc calls
  namespace math {

#if 0
    template <typename T, enable_if_t<is_floating_point_v<T>> = 0>
    constexpr T sqrtNewtonRaphson(T x, T curr = 1, T prev = 0) noexcept {
      return curr == prev ? curr : sqrtNewtonRaphson(x, (T)0.5 * (curr + x / curr), curr);
    }
#else
    template <typename T, enable_if_t<is_floating_point_v<T>> = 0>
    constexpr T sqrtNewtonRaphson(T n, T relTol = (T)(sizeof(T) > 4 ? 1e-9 : 1e-6)) noexcept {
      constexpr auto eps = (T)128 * zs::detail::deduce_numeric_epsilon<T>();
      if (n < -eps) return (T)limits<T>::quiet_NaN();
      if (n < (T)eps) return (T)0;

      T xn = (T)1;
      T xnp1 = (T)0.5 * (xn + n / xn);
      for (const auto tol = max(n * relTol, eps); abs(xnp1 - xn) > tol;
           xnp1 = (T)0.5 * (xn + n / xn))
        xn = xnp1;
      return xnp1;
    }
#endif
    /// ref: http://www.lomont.org/papers/2003/InvSqrt.pdf
    /// ref: https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
    /// ref: https://community.wolfram.com/groups/-/m/t/1108896
    /// ref:
    /// https://www.codeproject.com/Articles/69941/Best-Square-Root-Method-Algorithm-Function-Precisi
    constexpr float q_rsqrt(float number) noexcept {
      u32 i{};
      float x2 = number * 0.5f, y = number;
      // i = *(uint32_t *)&y;
      i = reinterpret_bits<u32>(y);
      i = 0x5f375a86 - (i >> 1);
      // y = *(float *)&i;
      y = reinterpret_bits<float>(i);
      y = y * (1.5f - (x2 * y * y));
      y = y * (1.5f - (x2 * y * y));
      return y;
    }
    constexpr double q_rsqrt(double number) noexcept {
      u64 i{};
      double x2 = number * 0.5, y = number;
      // i = *(uint64_t *)&y;
      i = reinterpret_bits<u64>(y);
      i = 0x5fe6eb50c7b537a9 - (i >> 1);
      // y = *(double *)&i;
      y = reinterpret_bits<double>(i);
      y = y * (1.5 - (x2 * y * y));
      y = y * (1.5 - (x2 * y * y));
      return y;
    }
    constexpr float q_sqrt(float x) noexcept { return 1.f / q_rsqrt(x); }
    // best guess starting square
    constexpr double q_sqrt(double fp) noexcept { return 1.0 / q_rsqrt(fp); }
    /// ref:
    /// https://stackoverflow.com/questions/66752842/ieee-754-conformant-sqrtf-implementation-taking-into-account-hardware-restrict
    /* square root computation suitable for all IEEE-754 binary32 arguments */
    constexpr float sqrt(float arg) noexcept {
      const float FP32_INFINITY = reinterpret_bits<float>(0x7f800000u);
      const float FP32_QNAN = reinterpret_bits<float>(0x7fffffffu); /* system specific */
      const float scale_in = 0x1.0p+26f;
      const float scale_out = 0x1.0p-13f;
      float rsq{}, err{}, sqt{};

      if (arg < 0.0f) {
        return FP32_QNAN;
      }
      // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g1c6fe34b4ac091e40eceeb0bae58459f
      else if ((arg == 0.0f) || !(abs(arg) < FP32_INFINITY)) { /* Inf, NaN */
        return arg + arg;
      } else {
        /* scale subnormal arguments towards unity */
        arg = arg * scale_in;

        /* generate low-accuracy approximation to rsqrt(arg) */
        rsq = q_rsqrt(arg);

        /* apply two Newton-Raphson iterations with quadratic convergence */
        rsq = ((-0.5f * arg * rsq) * rsq + 0.5f) * rsq + rsq;
        rsq = ((-0.5f * arg * rsq) * rsq + 0.5f) * rsq + rsq;

        /* compute sqrt from rsqrt, round to nearest or even */
        sqt = rsq * arg;
        err = sqt * -sqt + arg;
        sqt = (0.5f * rsq * err + sqt);

        /* compensate scaling of argument by counter-scaling the result */
        sqt = sqt * scale_out;

        return sqt;
      }
    }

    /// ref: codim-IPC
    template <typename T> constexpr T get_smallest_positive_real_quad_root(T a, T b, T c, T tol) {
      // return negative value if no positive real root is found
      T t{};
      if (zs::abs(a) <= tol)
        t = -c / b;
      else {
        double desc = b * b - 4 * a * c;
        if (desc > 0) {
          t = (-b - zs::sqrt(desc)) / (2 * a);
          if (t < 0) t = (-b + zs::sqrt(desc)) / (2 * a);
        } else  // desv<0 ==> imag
          t = -1;
      }
      return t;
    }
    template <typename T>
    constexpr T get_smallest_positive_real_cubic_root(T a, T b, T c, T d, T tol) {
      // return negative value if no positive real root is found
      T t = -1;
      if (zs::abs(a) <= tol)
        t = get_smallest_positive_real_quad_root(b, c, d, tol);
      else {
        zs::complex<T> i(0, 1);
        zs::complex<T> delta0(b * b - 3 * a * c, 0);
        zs::complex<T> delta1(2 * b * b * b - 9 * a * b * c + 27 * a * a * d, 0);
        zs::complex<T> C = zs::pow(
            (delta1 + zs::sqrt(delta1 * delta1 - (T)4.0 * delta0 * delta0 * delta0)) / (T)2.0,
            (T)1.0 / (T)3.0);
        if (zs::abs(C) == (T)0.0) {
          // a corner case listed by wikipedia found by our collaborate from another project
          C = zs::pow(
              (delta1 - zs::sqrt(delta1 * delta1 - (T)4.0 * delta0 * delta0 * delta0)) / (T)2.0,
              (T)1.0 / (T)3.0);
        }
        zs::complex<T> u2 = ((T)-1.0 + zs::sqrt((T)3.0) * i) / (T)2.0;
        zs::complex<T> u3 = ((T)-1.0 - zs::sqrt((T)3.0) * i) / (T)2.0;
        zs::complex<T> t1 = (b + C + delta0 / C) / ((T)-3.0 * a);
        zs::complex<T> t2 = (b + u2 * C + delta0 / (u2 * C)) / ((T)-3.0 * a);
        zs::complex<T> t3 = (b + u3 * C + delta0 / (u3 * C)) / ((T)-3.0 * a);

        if ((zs::abs(imag(t1)) < tol) && (real(t1) > 0)) t = real(t1);
        if ((zs::abs(imag(t2)) < tol) && (real(t2) > 0) && ((real(t2) < t) || (t < 0)))
          t = real(t2);
        if ((zs::abs(imag(t3)) < tol) && (real(t3) > 0) && ((real(t3) < t) || (t < 0)))
          t = real(t3);
      }
      return t;
    }

    template <typename T> constexpr T newton_solve_for_cubic_equation(T a, T b, T c, T d,
                                                                      T *results, int &numSols,
                                                                      T eps) {
      const auto __f
          = [](T x, T a, T b, T c, T d) { return a * x * x * x + b * x * x + c * x + d; };
      const auto __df = [](T x, T a, T b, T c) { return 3 * a * x * x + 2 * b * x + c; };
      T DX = 0;
      numSols = 0;
      T specialPoint = -b / a / 3;
      T pos[2] = {};
      int solves = 1;
      T delta = 4 * b * b - 12 * a * c;
      if (delta > 0) {
        pos[0] = (zs::sqrt(delta) - 2 * b) / 6 / a;
        pos[1] = (-zs::sqrt(delta) - 2 * b) / 6 / a;
        T v1 = __f(pos[0], a, b, c, d);
        T v2 = __f(pos[1], a, b, c, d);
        if (zs::abs(v1) < eps * eps) {
          v1 = 0;
        }
        if (zs::abs(v2) < eps * eps) {
          v2 = 0;
        }
        T sign = v1 * v2;
        DX = (pos[0] - pos[1]);
        if (sign <= 0) {
          solves = 3;
        } else if (sign > 0) {
          if ((a < 0 && __f(pos[0], a, b, c, d) > 0) || (a > 0 && __f(pos[0], a, b, c, d) < 0)) {
            DX = -DX;
          }
        }
      } else if (delta == 0) {
        if (zs::abs(__f(specialPoint, a, b, c, d)) < eps * eps) {
          for (int i = 0; i < 3; i++) {
            T tempReuslt = specialPoint;
            results[numSols] = tempReuslt;
            numSols++;
          }
          return;
        }
        if (a > 0) {
          if (__f(specialPoint, a, b, c, d) > 0) {
            DX = 1;
          } else if (__f(specialPoint, a, b, c, d) < 0) {
            DX = -1;
          }
        } else if (a < 0) {
          if (__f(specialPoint, a, b, c, d) > 0) {
            DX = -1;
          } else if (__f(specialPoint, a, b, c, d) < 0) {
            DX = 1;
          }
        }
      }

      T start = specialPoint - DX;
      T x0 = start;

      for (int i = 0; i < solves; i++) {
        T x1 = 0;
        int itCount = 0;
        do {
          if (itCount) x0 = x1;

          x1 = x0 - ((__f(x0, a, b, c, d)) / (__df(x0, a, b, c)));
          itCount++;
        } while (zs::abs(x1 - x0) > eps && itCount < 100000);
        results[numSols] = (x1);
        numSols++;
        start = start + DX;
        x0 = start;
      }
    }

  }  // namespace math

}  // namespace zs
