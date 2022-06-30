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
  //    int_type
  //    Nickolay Mladenov, for the implementation of operator+=

  struct rational {
    using int_type = i64;  // 128 would be better
    constexpr rational() noexcept : num{0}, den{1} {}
    constexpr rational(int_type n) noexcept : num{n}, den{1} {}
    constexpr rational(int_type n, int_type d) noexcept : num{n}, den{d} { canonicalize(); }
    template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
    constexpr rational(T v) noexcept {
      to_rational(v, num, den);
      if (den < 0) {
        num = -num;
        den = -den;
      }
    }
    constexpr rational(const rational &) noexcept = default;
    constexpr rational(rational &&) noexcept = default;
    constexpr rational &operator=(const rational &) noexcept = default;
    constexpr rational &operator=(rational &&) noexcept = default;

    constexpr int get_sign() const noexcept { return num > 0 ? 1 : (num < 0 ? -1 : 0); }
    constexpr void canonicalize() {
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
    constexpr int_type numerator() const noexcept { return num; }
    constexpr int_type denominator() const noexcept { return den; }
    std::string get_denominator_str() const { return std::to_string(denominator()); }
    std::string get_numerator_str() const { return std::to_string(numerator()); }

    constexpr double to_double() const noexcept { return (double)num / den; }
    static double get_double(const std::string &num, const std::string &denom) {
      rational v{std::stoll(num), std::stoll(denom)};
      return v.to_double();
    }

    //<<
    friend std::ostream &operator<<(std::ostream &os, const rational &r) {
      os << r.to_double();
      return os;
    }

    /// Arithmetic assignment operators
    constexpr rational &operator+=(const rational &r) {
      // This calculation avoids overflow, and minimises the number of expensive
      // calculations. Thanks to Nickolay Mladenov for this algorithm.
      //
      // Proof:
      // We have to compute a/b + c/d, where gcd(a,b)=1 and gcd(b,c)=1.
      // Let g = gcd(b,d), and b = b1*g, d=d1*g. Then gcd(b1,d1)=1
      //
      // The result is (a*d1 + c*b1) / (b1*d1*g).
      // Now we have to normalize this ratio.
      // Let's assume h | gcd((a*d1 + c*b1), (b1*d1*g)), and h > 1
      // If h | b1 then gcd(h,d1)=1 and hence h|(a*d1+c*b1) => h|a.
      // But since gcd(a,b1)=1 we have h=1.
      // Similarly h|d1 leads to h=1.
      // So we have that h | gcd((a*d1 + c*b1) , (b1*d1*g)) => h|g
      // Finally we have gcd((a*d1 + c*b1), (b1*d1*g)) = gcd((a*d1 + c*b1), g)
      // Which proves that instead of normalizing the result, it is better to
      // divide num and den by gcd((a*d1 + c*b1), g)

      // Protect against self-modification
      int_type r_num = r.num;
      int_type r_den = r.den;

      int_type g = math::gcd(den, r_den);
      den /= g;  // = b1 from the calculations above
      num = num * (r_den / g) + r_num * den;
      g = math::gcd(num, g);
      num /= g;
      den *= r_den / g;

      return *this;
    }

    constexpr rational &operator-=(const rational &r) {
      // Protect against self-modification
      int_type r_num = r.num;
      int_type r_den = r.den;

      // This calculation avoids overflow, and minimises the number of expensive
      // calculations. It corresponds exactly to the += case above
      int_type g = math::gcd(den, r_den);
      den /= g;
      num = num * (r_den / g) - r_num * den;
      g = math::gcd(num, g);
      num /= g;
      den *= r_den / g;

      return *this;
    }

    constexpr rational &operator*=(const rational &r) {
      // Protect against self-modification
      int_type r_num = r.num;
      int_type r_den = r.den;

      // Avoid overflow and preserve normalization
      int_type gcd1 = math::gcd(num, r_den);
      int_type gcd2 = math::gcd(r_num, den);
      num = (num / gcd1) * (r_num / gcd2);
      den = (den / gcd2) * (r_den / gcd1);
      return *this;
    }

    constexpr rational &operator/=(const rational &r) {
      // Protect against self-modification
      int_type r_num = r.num;
      int_type r_den = r.den;

      // Avoid repeated construction
      int_type zero(0);

      // Trap division by zero
      if (r_num == zero) {
        printf("rational division by zero!\n");
      }
      if (num == zero) return *this;

      // Avoid overflow and preserve normalization
      int_type gcd1 = math::gcd(num, r_num);
      int_type gcd2 = math::gcd(r_den, den);
      num = (num / gcd1) * (r_den / gcd2);
      den = (den / gcd2) * (r_num / gcd1);

      if (den < zero) {
        num = -num;
        den = -den;
      }
      return *this;
    }

    constexpr rational operator-() const noexcept { return rational(-num, den); }
    friend constexpr rational operator+(const rational &l, const rational &r) {
      rational tmp{l};
      return tmp += r;
    }
    friend constexpr rational operator-(const rational &l, const rational &r) {
      rational tmp{l};
      return tmp -= r;
    }
    friend constexpr rational operator*(const rational &l, const rational &r) {
      rational tmp{l};
      return tmp *= r;
    }
    friend constexpr rational operator/(const rational &l, const rational &r) {
      rational tmp{l};
      return tmp /= r;
    }

    /// Comparison operators
    friend constexpr bool operator<(const rational &l, const rational &r) {
      // Avoid repeated construction
      int_type const zero(0);

      // This should really be a class-wide invariant.  The reason for these
      // checks is that for 2's complement systems, INT_MIN has no corresponding
      // positive, so negating it during normalization keeps it INT_MIN, which
      // is bad for later calculations that assume a positive denominator.
      // BOOST_ASSERT(this->den > zero);
      // BOOST_ASSERT(r.den > zero);

      // Determine relative order by expanding each value to its simple continued
      // fraction representation using the Euclidian GCD algorithm.
      struct {
        int_type n, d, q, r;
      } ts = {l.num, l.den, static_cast<int_type>(l.num / l.den),
              static_cast<int_type>(l.num % l.den)},
        rs = {r.num, r.den, static_cast<int_type>(r.num / r.den),
              static_cast<int_type>(r.num % r.den)};
      unsigned reverse = 0u;

      // Normalize negative moduli by repeatedly adding the (positive) denominator
      // and decrementing the quotient.  Later cycles should have all positive
      // values, so this only has to be done for the first cycle.  (The rules of
      // C++ require a nonnegative quotient & remainder for a nonnegative dividend
      // & positive divisor.)
      while (ts.r < zero) {
        ts.r += ts.d;
        --ts.q;
      }
      while (rs.r < zero) {
        rs.r += rs.d;
        --rs.q;
      }

      // Loop through and compare each variable's continued-fraction components
      while (true) {
        // The quotients of the current cycle are the continued-fraction
        // components.  Comparing two c.f. is comparing their sequences,
        // stopping at the first difference.
        if (ts.q != rs.q) {
          // Since reciprocation changes the relative order of two variables,
          // and c.f. use reciprocals, the less/greater-than test reverses
          // after each index.  (Start w/ non-reversed @ whole-number place.)
          return reverse ? ts.q > rs.q : ts.q < rs.q;
        }

        // Prepare the next cycle
        reverse ^= 1u;

        if ((ts.r == zero) || (rs.r == zero)) {
          // At least one variable's c.f. expansion has ended
          break;
        }

        ts.n = ts.d;
        ts.d = ts.r;
        ts.q = ts.n / ts.d;
        ts.r = ts.n % ts.d;
        rs.n = rs.d;
        rs.d = rs.r;
        rs.q = rs.n / rs.d;
        rs.r = rs.n % rs.d;
      }

      // Compare infinity-valued components for otherwise equal sequences
      if (ts.r == rs.r) {
        // Both remainders are zero, so the next (and subsequent) c.f.
        // components for both sequences are infinity.  Therefore, the sequences
        // and their corresponding values are equal.
        return false;
      } else {
        // Exactly one of the remainders is zero, so all following c.f.
        // components of that variable are infinity, while the other variable
        // has a finite next c.f. component.  So that other variable has the
        // lesser value (modulo the reversal flag!).
        return (ts.r != zero) != static_cast<bool>(reverse);
      }
    }
    friend constexpr bool operator==(const rational &l, const rational &r) noexcept {
      return l.num == r.num && l.den == r.den;
    }
    friend constexpr bool operator!=(const rational &l, const rational &r) noexcept {
      return !(l == r);
    }
    friend constexpr bool operator>(const rational &l, const rational &r) {
      if (l == r) return false;
      return !(l < r);
    }

    int_type num, den;
  };

  ///
  /// ref: Tight-Inclusion
  ///
  // calculate 2^exponent
  constexpr u64 pow2(const u8 exponent) noexcept { return (u64)1l << exponent; }
  // return power t. n=result*2^t
  constexpr u8 reduction(const u64 n, u64 &result) noexcept {
    u8 t = 0;
    result = n;
    while (result != 0 && (result & 1) == 0) {
      result >>= 1;
      t++;
    }
    return t;
  }

  static constexpr u8 MAX_DENOM_POWER = 8 * sizeof(u64) - 1;
  //<k,n> pair present a number k/pow(2,n)
  struct NumCCD {
    u64 numerator;
    u8 denom_power;

    NumCCD() noexcept = default;
    ~NumCCD() noexcept = default;
    constexpr NumCCD(const NumCCD &) noexcept = default;
    constexpr NumCCD(NumCCD &&) noexcept = default;
    constexpr NumCCD &operator=(const NumCCD &) noexcept = default;
    constexpr NumCCD &operator=(NumCCD &&) noexcept = default;

    constexpr NumCCD(u64 p_numerator, u8 p_denom_power)
        : numerator{p_numerator}, denom_power{p_denom_power} {}

    constexpr NumCCD(double x) noexcept : numerator{}, denom_power{} {
      NumCCD low{0, 0}, high{1, 0}, mid{};

      // Hard code these cases for better accuracy.
      if (x == 0) {
        *this = low;
        return;
      } else if (x == 1) {
        *this = high;
        return;
      }

      do {
        mid = low + high;
        mid.denom_power++;

        if (mid.denom_power >= MAX_DENOM_POWER) {
          break;
        }

        if (x > mid) {
          low = mid;
        } else if (x < mid) {
          high = mid;
        } else {
          break;
        }
      } while (mid.denom_power < MAX_DENOM_POWER);
      *this = high;
    }

    constexpr u64 denominator() const noexcept { return (u64)1 << denom_power; }

    // convert NumCCD to double number
    constexpr double value() const noexcept { return (double)numerator / denominator(); }

    constexpr operator double() const noexcept { return value(); }

    constexpr NumCCD operator+(const NumCCD &other) const noexcept {
      const u64 &k1 = numerator, &k2 = other.numerator;
      const u8 &n1 = denom_power, &n2 = other.denom_power;

      NumCCD result{};
      if (n1 == n2) {
        result.denom_power = n2 - reduction(k1 + k2, result.numerator);
      } else if (n2 > n1) {
        result.numerator = k1 * pow2(n2 - n1) + k2;
        // assert(result.numerator % 2 == 1);
        result.denom_power = n2;
      } else {  // n2 < n1
        result.numerator = k1 + k2 * pow2(n1 - n2);
        // assert(result.numerator % 2 == 1);
        result.denom_power = n1;
      }
      return result;
    }

    constexpr bool operator==(const NumCCD &other) const noexcept {
      return numerator == other.numerator && denom_power == other.denom_power;
    }
    constexpr bool operator!=(const NumCCD &other) const { return !(*this == other); }
    constexpr bool operator<(const NumCCD &other) const {
      const u64 &k1 = numerator, &k2 = other.numerator;
      const u8 &n1 = denom_power, &n2 = other.denom_power;

      u64 tmp_k1 = k1, tmp_k2 = k2;
      if (n1 < n2) {
        tmp_k1 = pow2(n2 - n1) * k1;
      } else if (n1 > n2) {
        tmp_k2 = pow2(n1 - n2) * k2;
      }
      // assert((value() < other.value()) == (tmp_k1 < tmp_k2));
      return tmp_k1 < tmp_k2;
    }
    constexpr bool operator<=(const NumCCD &other) const {
      return (*this == other) || (*this < other);
    }
    constexpr bool operator>=(const NumCCD &other) const { return !(*this < other); }
    constexpr bool operator>(const NumCCD &other) const { return !(*this <= other); }

    constexpr bool operator<(const double other) const { return value() < other; }
    constexpr bool operator>(const double other) const { return value() > other; }
    constexpr bool operator==(const double other) const { return value() == other; }

    static constexpr bool is_sum_leq_1(const NumCCD &num1, const NumCCD &num2) {
      if (num1.denom_power == num2.denom_power) {
        // skip the reduction in num1 + num2
        return num1.numerator + num2.numerator <= num1.denominator();
      }
      NumCCD tmp = num1 + num2;
      return tmp.numerator <= tmp.denominator();
    }
  };

  // an interval represented by two double numbers
  struct Interval {
    NumCCD lower;
    NumCCD upper;

    Interval() noexcept = default;
    ~Interval() noexcept = default;
    constexpr Interval(const Interval &) noexcept = default;
    constexpr Interval(Interval &&) noexcept = default;
    constexpr Interval &operator=(const Interval &) noexcept = default;
    constexpr Interval &operator=(Interval &&) noexcept = default;

    constexpr Interval(const NumCCD &p_lower, const NumCCD &p_upper) noexcept
        : lower{p_lower}, upper{p_upper} {}

    constexpr zs::tuple<Interval, Interval> bisect() const noexcept {
      // interval is [k1/pow2(n1), k2/pow2(n2)]
      NumCCD mid = upper + lower;
      mid.denom_power++;  // ÷ 2
      // assert(mid.value() > lower.value() && mid.value() < upper.value());
      return zs::make_tuple(Interval(lower, mid), Interval(mid, upper));
    }

    constexpr bool overlaps(const double r1, const double r2) const noexcept {
      return upper.value() >= r1 && lower.value() <= r2;
    }
  };

  using Interval3 = zs::vec<Interval, 3>;
  using Array3 = zs::vec<double, 3>;

  constexpr Array3 width(const Interval3 &x) {
    return {x[0].upper.value() - x[0].lower.value(), x[1].upper.value() - x[1].lower.value(),
            x[2].upper.value() - x[2].lower.value()};
  }

  // find the largest width/tol dimension that is greater than its tolerance
  constexpr int find_next_split(const Array3 &widths, const Array3 &tols) noexcept {
    // assert((widths > tols).any());
    Array3 tmp = Array3::init([&widths, &tols](int d) {
      return widths[d] > tols[d] ? widths[d] / tols[d] : -std::numeric_limits<double>::infinity();
    });
    double val = tmp[0];
    int max_index{0};
    for (int i = 1; i != 3; ++i)
      if (tmp[i] > val) {
        val = tmp[i];
        max_index = i;
      }
    // tmp.maxCoeff(&max_index);
    return max_index;
  }

}  // namespace zs