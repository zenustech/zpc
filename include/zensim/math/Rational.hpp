#pragma once
#include "Vec.h"

namespace zs {

  /// ref:
  /// https://stackoverflow.com/questions/21211291/the-most-accurate-way-to-calculate-numerator-and-denominator-of-a-double
  template <typename T, typename Ti,
            zs::enable_if_all<std::is_floating_point_v<T>, std::is_integral_v<Ti>> = 0>
  constexpr void to_rational(T val, Ti &num, Ti &den) {
    if (zs::isnan(val)) {
      return;
    }
    if (zs::isinf(val)) {
      return;
    }

    T d{};

    if (zs::modf(val, &d) == 0) {
      // already a whole number
      num = val;
      den = 1;
      return;
    }

    int exponent{};
    T significand = zs::frexp(val, &exponent);  // val = significand * 2^exponent
    T numerator = val;
    T denominator = 1;

    // 0.5 <= significand < 1.0
    // significand is a fraction, multiply it by two until it's a whole number
    // subtract exponent appropriately to maintain val = significand * 2^exponent
    do {
      significand *= 2;
      --exponent;
      // assert(std::ldexp(significand, exponent) == val);
    } while (zs::modf(significand, &d) != 0);

    // assert(exponent <= 0);

    // significand is now a whole number
    if (significand < limits<Ti>::max() && significand > limits<Ti>::lowest())
      num = (Ti)significand;
    else
      printf("underlying integer not big enough!");
    if (auto v = (1.0 / zs::ldexp((T)1.0, exponent));
        v < limits<Ti>::max() && v > limits<Ti>::lowest())
      den = (Ti)v;
    else
      printf("underlying integer not big enough!");

    // assert(_val == _num / _den);
  }

  /// ref : https://www.boost.org/doc/libs/1_76_0/boost/rational.hpp
  //  Boost rational.hpp header file  ------------------------------------------//

  //  (C) Copyright Paul Moore 1999. Permission to copy, use, modify, sell and
  //  distribute this software is granted provided this copyright notice appears
  //  in all copies. This software is provided "as is" without express or
  //  implied warranty, and with no claim as to its suitability for any purpose.

  // boostinspect:nolicense (don't complain about the lack of a Boost license)
  // (Paul Moore hasn't been in contact for years, so there's no way to change the
  // license.)

  //  See http://www.boost.org/libs/rational for documentation.

  //  Credits:
  //  Thanks to the boost mailing list in general for useful comments.
  //  Particular contributions included:
  //    Andrew D Jewell, for reminding me to take care to avoid overflow
  //    Ed Brey, for many comments, including picking up on some dreadful typos
  //    Stephen Silver contributed the test suite and comments on user-defined
  //    IntType
  //    Nickolay Mladenov, for the implementation of operator+=

  struct rational {
    using int_type = i64;  // 128 would be better
    constexpr rational() noexcept : num{0}, den{1} {}
    constexpr rational(int_type n) noexcept : num{n}, den{1} {}
    constexpr rational(int_type n, int_type d) noexcept : num{n}, den{d} { normalize(); }
    template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
    constexpr rational(T v) noexcept {
      to_rational(v, num, den);
    }

    constexpr void normalize() {
      if (den == 0) {
        printf("invalid rational state!\n");
        return;
      }
      if (num == 0) {
        den = 1;
        return;
      }
      int_type g = math::gcd(num, den);

      num /= g;
      den /= g;
      if (den < 0) {
        num = -num;
        den = -den;
      }
    }

    int_type num, den;
  }

}  // namespace zs