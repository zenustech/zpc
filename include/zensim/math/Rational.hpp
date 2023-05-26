#pragma once
#include <iostream>
#include <string>

#include "Vec.h"

namespace zs {

  /// ref:
  /// https://stackoverflow.com/questions/21211291/the-most-accurate-way-to-calculate-numerator-and-denominator-of-a-double
  template <typename T, typename Ti,
            zs::enable_if_all<is_floating_point_v<T>, is_integral_v<Ti>> = 0>
  constexpr void to_rational(T val, Ti &num, Ti &den) {
    if (zs::isnan(val)) {
      return;
    }
#if 0
    if (zs::isinf(val)) {
      return;
    }
#endif

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
    if (significand < detail::deduce_numeric_max<Ti>()
        && significand > detail::deduce_numeric_lowest<Ti>())
      num = (Ti)significand;
    else
      printf("underlying integer not big enough!");
    if (auto v = (1.0 / zs::ldexp((T)1.0, exponent));
        v < detail::deduce_numeric_max<Ti>() && v > detail::deduce_numeric_lowest<Ti>())
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
    template <typename T, enable_if_t<is_floating_point_v<T>> = 0>
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
  /// https://github.com/Continuous-Collision-Detection/Tight-Inclusion
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

  constexpr zs::vec<double, 8> function_ee(
      const double &a0s, const double &a1s, const double &b0s, const double &b1s, const double &a0e,
      const double &a1e, const double &b0e, const double &b1e, const zs::vec<double, 8> &t_up,
      const zs::vec<double, 8> &t_dw, const zs::vec<double, 8> &u_up,
      const zs::vec<double, 8> &u_dw, const zs::vec<double, 8> &v_up,
      const zs::vec<double, 8> &v_dw) {
    zs::vec<double, 8> rst{};
    for (int i = 0; i < 8; i++) {
      double edge0_vertex0 = (a0e - a0s) * t_up[i] / t_dw[i] + a0s;
      double edge0_vertex1 = (a1e - a1s) * t_up[i] / t_dw[i] + a1s;
      double edge1_vertex0 = (b0e - b0s) * t_up[i] / t_dw[i] + b0s;
      double edge1_vertex1 = (b1e - b1s) * t_up[i] / t_dw[i] + b1s;

      double edge0_vertex = (edge0_vertex1 - edge0_vertex0) * u_up[i] / u_dw[i] + edge0_vertex0;
      double edge1_vertex = (edge1_vertex1 - edge1_vertex0) * v_up[i] / v_dw[i] + edge1_vertex0;
      rst[i] = edge0_vertex - edge1_vertex;
    }
    return rst;
  }

  constexpr zs::vec<double, 8> function_vf(
      const double &vs, const double &t0s, const double &t1s, const double &t2s, const double &ve,
      const double &t0e, const double &t1e, const double &t2e, const zs::vec<double, 8> &t_up,
      const zs::vec<double, 8> &t_dw, const zs::vec<double, 8> &u_up,
      const zs::vec<double, 8> &u_dw, const zs::vec<double, 8> &v_up,
      const zs::vec<double, 8> &v_dw) {
    zs::vec<double, 8> rst{};
    for (int i = 0; i < 8; i++) {
      double v = (ve - vs) * t_up[i] / t_dw[i] + vs;
      double t0 = (t0e - t0s) * t_up[i] / t_dw[i] + t0s;
      double t1 = (t1e - t1s) * t_up[i] / t_dw[i] + t1s;
      double t2 = (t2e - t2s) * t_up[i] / t_dw[i] + t2s;
      double pt = (t1 - t0) * u_up[i] / u_dw[i] + (t2 - t0) * v_up[i] / v_dw[i] + t0;
      rst[i] = v - pt;
    }
    return rst;
  }

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

  constexpr void convert_tuv_to_array(const Interval3 &itv, zs::vec<double, 8> &t_up,
                                      zs::vec<double, 8> &t_dw, zs::vec<double, 8> &u_up,
                                      zs::vec<double, 8> &u_dw, zs::vec<double, 8> &v_up,
                                      zs::vec<double, 8> &v_dw) {
    // t order: 0,0,0,0,1,1,1,1
    // u order: 0,0,1,1,0,0,1,1
    // v order: 0,1,0,1,0,1,0,1
    const double t0_up = itv[0].lower.numerator, t0_dw = itv[0].lower.denominator(),
                 t1_up = itv[0].upper.numerator, t1_dw = itv[0].upper.denominator(),
                 u0_up = itv[1].lower.numerator, u0_dw = itv[1].lower.denominator(),
                 u1_up = itv[1].upper.numerator, u1_dw = itv[1].upper.denominator(),
                 v0_up = itv[2].lower.numerator, v0_dw = itv[2].lower.denominator(),
                 v1_up = itv[2].upper.numerator, v1_dw = itv[2].upper.denominator();
    t_up = {t0_up, t0_up, t0_up, t0_up, t1_up, t1_up, t1_up, t1_up};
    t_dw = {t0_dw, t0_dw, t0_dw, t0_dw, t1_dw, t1_dw, t1_dw, t1_dw};
    u_up = {u0_up, u0_up, u1_up, u1_up, u0_up, u0_up, u1_up, u1_up};
    u_dw = {u0_dw, u0_dw, u1_dw, u1_dw, u0_dw, u0_dw, u1_dw, u1_dw};
    v_up = {v0_up, v1_up, v0_up, v1_up, v0_up, v1_up, v0_up, v1_up};
    v_dw = {v0_dw, v1_dw, v0_dw, v1_dw, v0_dw, v1_dw, v0_dw, v1_dw};
  }
  template <typename T, int N>
  constexpr void min_max_array(const zs::vec<T, N> &arr, T &min, T &max) {
    static_assert(N > 0, "no min/max of empty array");
    min = arr[0];
    max = arr[0];
    for (int i = 1; i != N; ++i) {
      if (min > arr[i]) {
        min = arr[i];
      }
      if (max < arr[i]) {
        max = arr[i];
      }
    }
  }

  // ** this version can return the true x or y or z tolerance of the co-domain **
  // eps is the interval [-eps,eps] we need to check
  // if [-eps,eps] overlap, return true
  // bbox_in_eps tell us if the box is totally in eps box
  // ms is the minimum seperation
  template <bool check_vf> constexpr bool evaluate_bbox_one_dimension_vector(
      zs::vec<double, 8> &t_up, zs::vec<double, 8> &t_dw, zs::vec<double, 8> &u_up,
      zs::vec<double, 8> &u_dw, zs::vec<double, 8> &v_up, zs::vec<double, 8> &v_dw,
      const Array3 &a0s, const Array3 &a1s, const Array3 &b0s, const Array3 &b1s, const Array3 &a0e,
      const Array3 &a1e, const Array3 &b0e, const Array3 &b1e, const int dim, const double eps,
      bool &bbox_in_eps, const double ms = 0, double *tol = nullptr) {
    zs::vec<double, 8> vs{};
    if constexpr (check_vf) {
      vs = function_vf(a0s[dim], a1s[dim], b0s[dim], b1s[dim], a0e[dim], a1e[dim], b0e[dim],
                       b1e[dim], t_up, t_dw, u_up, u_dw, v_up, v_dw);
    } else {
      vs = function_ee(a0s[dim], a1s[dim], b0s[dim], b1s[dim], a0e[dim], a1e[dim], b0e[dim],
                       b1e[dim], t_up, t_dw, u_up, u_dw, v_up, v_dw);
    }

    double minv{}, maxv{};
    min_max_array<double, 8>(vs, minv, maxv);

    if (tol != nullptr) {
      *tol = maxv - minv;  // this is the real tolerance
    }

    bbox_in_eps = false;

    const double eps_and_ms = eps + ms;

    if (minv > eps_and_ms || maxv < -eps_and_ms) {
      return false;
    }

    if (minv >= -eps_and_ms && maxv <= eps_and_ms) {
      bbox_in_eps = true;
    }

    return true;
  }
  // ** this version can return the true tolerance of the co-domain **
  // give the result of if the hex overlaps the input eps box around origin
  // use vectorized hex-vertex-solving function for acceleration
  // box_in_eps shows if this hex is totally inside box. if so, no need to do further bisection
  template <bool check_vf> constexpr bool origin_in_function_bounding_box_vector(
      const Interval3 &paras, const Array3 &a0s, const Array3 &a1s, const Array3 &b0s,
      const Array3 &b1s, const Array3 &a0e, const Array3 &a1e, const Array3 &b0e, const Array3 &b1e,
      const Array3 &eps, bool &box_in_eps, const double ms = 0, Array3 *tolerance = nullptr) {
    box_in_eps = false;

    zs::vec<double, 8> t_up{}, t_dw{}, u_up{}, u_dw{}, v_up{}, v_dw{};
    convert_tuv_to_array(paras, t_up, t_dw, u_up, u_dw, v_up, v_dw);

    bool box_in[3] = {};
    for (int i = 0; i < 3; i++) {
      double *tol = tolerance == nullptr ? nullptr : &((*tolerance)[i]);
      if (!evaluate_bbox_one_dimension_vector<check_vf>(t_up, t_dw, u_up, u_dw, v_up, v_dw, a0s,
                                                        a1s, b0s, b1s, a0e, a1e, b0e, b1e, i,
                                                        eps[i], box_in[i], ms, tol)) {
        return false;
      }
    }

    if (box_in[0] && box_in[1] && box_in[2]) {
      box_in_eps = true;
    }

    return true;
  }
  template <typename F>  // std::function<void(const Interval3 &)>
  constexpr bool split_and_push(const Interval3 &tuv, int split_i, F &&push, bool check_vf,
                                double t_upper_bound = 1) {
    auto [halves_first, halves_second] = tuv[split_i].bisect();
    if (halves_first.lower >= halves_first.upper || halves_second.lower >= halves_second.upper) {
      printf("OVERFLOW HAPPENS WHEN SPLITTING INTERVALS");
      return true;
    }

    Interval3 tmp = tuv;

    if (split_i == 0) {
      if (t_upper_bound == 1 || halves_second.overlaps(0, t_upper_bound)) {
        tmp[split_i] = halves_second;
        push(tmp);
      }
      if (t_upper_bound == 1 || halves_first.overlaps(0, t_upper_bound)) {
        tmp[split_i] = halves_first;
        push(tmp);
      }
    } else if (!check_vf) {  // edge uv
      tmp[split_i] = halves_second;
      push(tmp);
      tmp[split_i] = halves_first;
      push(tmp);
    } else {
      // assert(check_vf && split_i != 0);
      // u + v ≤ 1
      if (split_i == 1) {
        const Interval &v = tuv[2];
        if (NumCCD::is_sum_leq_1(halves_second.lower, v.lower)) {
          tmp[split_i] = halves_second;
          push(tmp);
        }
        if (NumCCD::is_sum_leq_1(halves_first.lower, v.lower)) {
          tmp[split_i] = halves_first;
          push(tmp);
        }
      } else if (split_i == 2) {
        const Interval &u = tuv[1];
        if (NumCCD::is_sum_leq_1(u.lower, halves_second.lower)) {
          tmp[split_i] = halves_second;
          push(tmp);
        }
        if (NumCCD::is_sum_leq_1(u.lower, halves_first.lower)) {
          tmp[split_i] = halves_first;
          push(tmp);
        }
      }
    }
    return false;  // no overflow
  }

#define ZS_MAXIMUM_LOCAL_STACK_SIZE 256
  // this version cannot give the impact time at t=1, although this collision can
  // be detected at t=0 of the next time step, but still may cause problems in
  // line-search based physical simulation
  template <bool check_vf>
  constexpr bool interval_root_finder_DFS(const Array3 &a0s, const Array3 &a1s, const Array3 &b0s,
                                          const Array3 &b1s, const Array3 &a0e, const Array3 &a1e,
                                          const Array3 &b0e, const Array3 &b1e, const Array3 &tol,
                                          const Array3 &err, const double ms, double &toi) {
    auto cmp_time
        = [](const Interval3 &i1, const Interval3 &i2) { return i1[0].lower >= i2[0].lower; };

    // build interval set [0,1]x[0,1]x[0,1]
    const Interval zero_to_one = Interval(NumCCD(0, 0), NumCCD(1, 0));
    Interval3 iset{zero_to_one, zero_to_one, zero_to_one};

    // Stack of intervals and the last split dimension
    // std::stack<std::pair<Interval3,int>> istack;
    Interval3 istack[ZS_MAXIMUM_LOCAL_STACK_SIZE] = {iset};
    int top = 1;
    auto popPriority = [&istack, &top, &cmp_time]() {
      auto ret = istack[0];
      int id = 0;
      for (int i = 1; i < top; ++i)
        if (cmp_time(istack[i], ret)) {
          ret = istack[i];
          id = i;
        }
      --top;
      if (id != top) istack[id] = istack[top];
      return ret;
    };
    auto all_le = [](const Array3 &a, const Array3 &b) {
      return a[0] <= b[0] && a[1] <= b[1] && a[2] <= b[2];
    };

    Array3 err_and_ms = err + ms;

    int refine = 0;

    toi = detail::deduce_numeric_infinity<double>();
    NumCCD TOI(1, 0);

    bool collision = false;
    int rnbr = 0;
    while (top) {
      Interval3 current = popPriority();  // mimic priority queue: top + pop

      // TOI should always be no larger than current
      if (current[0].lower >= TOI) continue;

      refine++;

      bool zero_in{}, box_in{};
      zero_in = origin_in_function_bounding_box_vector<check_vf>(current, a0s, a1s, b0s, b1s, a0e,
                                                                 a1e, b0e, b1e, err_and_ms, box_in);

      if (!zero_in) continue;

      Array3 widths = width(current);

      if (box_in || all_le(widths, tol)) {
        TOI = current[0].lower;
        collision = true;
        rnbr++;
        // continue;
        toi = TOI.value();
        return true;
      }

      // find the next dimension to split
      int split_i = find_next_split(widths, tol);

      bool overflowed = split_and_push(
          current, split_i,
          [&](const Interval3 &i) {
            if (top < ZS_MAXIMUM_LOCAL_STACK_SIZE) istack[top++] = i;
          },
          check_vf);
      if (top > ZS_MAXIMUM_LOCAL_STACK_SIZE - 1) {
        printf("local stack depth not enough!\n");
        return true;
      }
      if (overflowed) {
        printf("OVERFLOW HAPPENS WHEN SPLITTING INTERVALS");
        return true;
      }
    }
    // if (collision) toi = TOI.value();
    return collision;
  }

  // when check_t_overlap = false, check [0,1]x[0,1]x[0,1]; otherwise, check [0, t_max]x[0,1]x[0,1]
  template <bool check_vf> constexpr bool interval_root_finder_BFS(
      const Array3 &a0s, const Array3 &a1s, const Array3 &b0s, const Array3 &b1s, const Array3 &a0e,
      const Array3 &a1e, const Array3 &b0e, const Array3 &b1e, const Array3 &tol,
      const double co_domain_tolerance, const Array3 &err, const double ms, const double max_time,
      const int max_itr, double &toi, double &output_tolerance, bool &earlyTerminate) {
    // if max_itr <0, output_tolerance= co_domain_tolerance;
    // else, output_tolearancewill be the precision after iteration time > max_itr
    output_tolerance = co_domain_tolerance;

    // this is used to catch the tolerance for each level
    double temp_output_tolerance = co_domain_tolerance;

    using Record = zs::tuple<Interval3, int>;
    // check the tree level by level instead of going deep
    // (if level 1 != level 2, return level 1 >= level 2; else, return time1 >= time2)
    auto horiz_cmp = [](const Record &i1, const Record &i2) {
      if (zs::get<1>(i1) != zs::get<1>(i2)) {
        return zs::get<1>(i1) >= zs::get<1>(i2);
      } else {
        return zs::get<0>(i1)[0].lower > zs::get<0>(i2)[0].lower;
      }
    };

    // Stack of intervals and the last split dimension
    const Interval zero_to_one = Interval(NumCCD(0, 0), NumCCD(1, 0));
    Interval3 iset{zero_to_one, zero_to_one, zero_to_one};

    // Stack of intervals and the last split dimension
    Record istack[ZS_MAXIMUM_LOCAL_STACK_SIZE] = {zs::make_tuple(iset, -1)};
    int top = 1;
    auto popPriority = [&istack, &top, &cmp = horiz_cmp]() {
      auto ret = istack[0];
      int id = 0;
      for (int i = 1; i < top; ++i)
        if (cmp(istack[i], ret)) {
          ret = istack[i];
          id = i;
        }
      --top;
      if (id != top) istack[id] = istack[top];
      return ret;
    };
    auto all_le = [](const Array3 &a, const Array3 &b) {
      return a[0] <= b[0] && a[1] <= b[1] && a[2] <= b[2];
    };

    // current intervals
    int refine = 0;
    double impact_ratio = 1;

    toi = detail::deduce_numeric_infinity<double>();  // set toi as infinate
    // temp_toi is to catch the toi of each level
    double temp_toi = toi;
    // set TOI to 4. this is to record the impact time of this level
    NumCCD TOI(4, 0);
    // this is to record the element that already small enough or contained in eps-box
    NumCCD TOI_SKIP = TOI;
    bool use_skip = false;  // this is to record if TOI_SKIP is used.
    int rnbr = 0;
    int current_level = -2;  // in the begining, current_level != level
    int box_in_level = -2;   // this checks if all the boxes before this
    // level < tolerance. only true, we can return when we find one overlaps eps box and smaller
    // than tolerance or eps-box
    bool this_level_less_tol = true;
    bool find_level_root = false;
    double t_upper_bound = max_time;  // 2*tol make it more conservative
    while (top) {
      auto [current, level] = popPriority();

      // if this box is later than TOI_SKIP in time, we can skip this one.
      // TOI_SKIP is only updated when the box is small enough or totally contained in eps-box
      if (current[0].lower >= TOI_SKIP) {
        continue;
      }
      // before check a new level, set this_level_less_tol=true
      if (box_in_level != level) {
        box_in_level = level;
        this_level_less_tol = true;
      }

      refine++;
      bool zero_in{}, box_in{};
      Array3 true_tol{};
      zero_in = origin_in_function_bounding_box_vector<check_vf>(
          current, a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e, err, box_in, ms, &true_tol);

      if (!zero_in) continue;

      Array3 widths = width(current);

      bool tol_condition = all_le(true_tol, co_domain_tolerance);

      // Condition 1, stopping condition on t, u and v is satisfied. this is useless now since we
      // have condition 2
      bool condition1 = all_le(widths, tol);

      // Condition 2, zero_in = true, box inside eps-box and in this level,
      // no box whose zero_in is true but box size larger than tolerance, can return
      bool condition2 = box_in && this_level_less_tol;
      if (!tol_condition) {
        this_level_less_tol = false;
        // this level has at least one box whose size > tolerance, thus we
        // cannot directly return if find one box whose size < tolerance or box-in
        // TODO: think about it. maybe we can return even if this value is false, so we can
        // terminate earlier.
      }

      // Condition 3, in this level, we find a box that zero-in and size < tolerance.
      // and no other boxes whose zero-in is true in this level before this one is larger than
      // tolerance, can return
      bool condition3 = this_level_less_tol;
      if (condition1 || condition2 || condition3) {
        TOI = current[0].lower;
        rnbr++;
        // continue;
        toi = TOI.value() * impact_ratio;
        // we don't need to compare with TOI_SKIP because we already
        // continue when t >= TOI_SKIP
        return true;
      }

      if (max_itr > 0) {  // if max_itr <= 0 ⟹ unlimited iterations
        if (current_level != level) {
          // output_tolerance=current_tolerance;
          // current_tolerance=0;
          current_level = level;
          find_level_root = false;
        }
        // current_tolerance=std::max(
        // std::max(std::max(current_tolerance,true_tol[0]),true_tol[1]),true_tol[2]
        // );
        if (!find_level_root) {
          TOI = current[0].lower;
          // collision=true;
          rnbr++;
          // continue;
          temp_toi = TOI.value() * impact_ratio;

          // if the real tolerance is larger than input, use the real one;
          // if the real tolerance is smaller than input, use input
          temp_output_tolerance = zs::max(zs::max(true_tol[0], true_tol[1]),
                                          zs::max(true_tol[2], co_domain_tolerance));
          // this ensures always find the earlist root
          find_level_root = true;
        }
        if (refine > max_itr) {
          toi = temp_toi;
          output_tolerance = temp_output_tolerance;
          earlyTerminate = true;

          // std::cout<<"return from refine"<<std::endl;
          return true;
        }
        // get the time of impact down here
      }

      // if this box is small enough, or inside of eps-box, then just continue,
      // but we need to record the collision time
      if (tol_condition || box_in) {
        if (current[0].lower < TOI_SKIP) {
          TOI_SKIP = current[0].lower;
        }
        use_skip = true;
        continue;
      }

      // find the next dimension to split
      int split_i = find_next_split(widths, tol);

      bool overflow = split_and_push(
          current, split_i,
          [&, level = level](const Interval3 &i) {
            if (top < ZS_MAXIMUM_LOCAL_STACK_SIZE) istack[top++] = zs::make_tuple(i, level + 1);
          },
          check_vf, t_upper_bound);
      if (top > ZS_MAXIMUM_LOCAL_STACK_SIZE - 1) {
        earlyTerminate = true;
        printf("local queue size not enough!\n");
        return true;
      }
      if (overflow) {
        printf("OVERFLOW HAPPENS WHEN SPLITTING INTERVALS");
        return true;
      }
    }

    if (use_skip) {
      toi = TOI_SKIP.value() * impact_ratio;
      return true;
    }

    return false;
  }

  template <int N> constexpr Array3 get_numerical_error(const zs::vec<Array3, N> &vertices,
                                                        const bool check_vf,
                                                        const bool using_minimum_separation) {
    double eefilter{};
    double vffilter{};
    if (!using_minimum_separation) {
      eefilter = 6.217248937900877e-15;
      vffilter = 6.661338147750939e-15;
    } else  // using minimum separation
    {
      eefilter = 7.105427357601002e-15;
      vffilter = 7.549516567451064e-15;
    }
    double filter = check_vf ? vffilter : eefilter;

    Array3 max = vertices[0].abs();
    for (int i = 0; i < N; ++i) {
      // max = max.cwiseMax(vertices[i].cwiseAbs());
      for (int d = 0; d != 3; ++d)
        if (vertices[i][d] > max[d]) max[d] = vertices[i][d];
    }
    Array3 delta = max.min(1);  // (..., 1]
    return filter
           * Array3{delta[0] * delta[0] * delta[0], delta[1] * delta[1] * delta[1],
                    delta[2] * delta[2] * delta[2]};
  }

  constexpr double max_linf_4(const Array3 &p1, const Array3 &p2, const Array3 &p3,
                              const Array3 &p4, const Array3 &p1e, const Array3 &p2e,
                              const Array3 &p3e, const Array3 &p4e) {
    return zs::max(zs::max((p1e - p1).infNorm(), (p2e - p2).infNorm()),
                   zs::max((p3e - p3).infNorm(), (p4e - p4).infNorm()));
  }

#define CCD_MAX_TIME_TOL 1e-3
#define CCD_MAX_COORD_TOL 1e-2
  constexpr Array3 compute_edge_edge_tolerances(
      const Array3 &edge0_vertex0_start, const Array3 &edge0_vertex1_start,
      const Array3 &edge1_vertex0_start, const Array3 &edge1_vertex1_start,
      const Array3 &edge0_vertex0_end, const Array3 &edge0_vertex1_end,
      const Array3 &edge1_vertex0_end, const Array3 &edge1_vertex1_end,
      const double distance_tolerance) {
    const Array3 p000 = edge0_vertex0_start - edge1_vertex0_start;
    const Array3 p001 = edge0_vertex0_start - edge1_vertex1_start;
    const Array3 p011 = edge0_vertex1_start - edge1_vertex1_start;
    const Array3 p010 = edge0_vertex1_start - edge1_vertex0_start;
    const Array3 p100 = edge0_vertex0_end - edge1_vertex0_end;
    const Array3 p101 = edge0_vertex0_end - edge1_vertex1_end;
    const Array3 p111 = edge0_vertex1_end - edge1_vertex1_end;
    const Array3 p110 = edge0_vertex1_end - edge1_vertex0_end;

    double dl = 3 * max_linf_4(p000, p001, p011, p010, p100, p101, p111, p110);
    double edge0_length = 3 * max_linf_4(p000, p100, p101, p001, p010, p110, p111, p011);
    double edge1_length = 3 * max_linf_4(p000, p100, p110, p010, p001, p101, p111, p011);

    return Array3(zs::min(distance_tolerance / dl, CCD_MAX_TIME_TOL),
                  zs::min(distance_tolerance / edge0_length, CCD_MAX_COORD_TOL),
                  zs::min(distance_tolerance / edge1_length, CCD_MAX_COORD_TOL));
  }
  constexpr bool edgeEdgeCCD(const Array3 &a0s, const Array3 &a1s, const Array3 &b0s,
                             const Array3 &b1s, const Array3 &a0e, const Array3 &a1e,
                             const Array3 &b0e, const Array3 &b1e, const Array3 &err_in,
                             const double ms_in, double &toi, const double tolerance_in,
                             const double t_max_in, const int max_itr, double &output_tolerance,
                             bool &earlyTerminate, bool no_zero_toi = true) {
    constexpr int MAX_NO_ZERO_TOI_ITER = detail::deduce_numeric_max<int>();
    // unsigned so can be larger than MAX_NO_ZERO_TOI_ITER
    unsigned int no_zero_toi_iter = 0;

    earlyTerminate = false;
    bool is_impacting{}, tmp_is_impacting{};

    // Mutable copies for no_zero_toi
    double t_max = t_max_in;
    double tolerance = tolerance_in;
    double ms = ms_in;

    Array3 tol = compute_edge_edge_tolerances(a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e, tolerance_in);

    //////////////////////////////////////////////////////////
    // this should be the error of the whole mesh
    Array3 err{};
    // if error[0]<0, means we need to calculate error here
    if (err_in[0] < 0) {
      zs::vec<Array3, 8> vlist = {a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e};
      bool use_ms = ms > 0;
      err = get_numerical_error(vlist, false, use_ms);
    } else {
      err = err_in;
    }
    //////////////////////////////////////////////////////////

    do {
#if 1
      // case CCDRootFindingMethod::DEPTH_FIRST_SEARCH:
      // no handling for zero toi
      return interval_root_finder_DFS<false>(a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e, tol, err, ms,
                                             toi);
#else
      tmp_is_impacting = interval_root_finder_BFS<false>(a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e,
                                                         tol, tolerance, err, ms, t_max, max_itr,
                                                         toi, output_tolerance, earlyTerminate);
      // if (tmp_is_impacting) return true;
#endif
      // assert(!tmp_is_impacting || toi >= 0);

      if (t_max == t_max_in) {
        // This will be the final output because we might need to
        // perform CCD again if the toi is zero. In which case we will
        // use a smaller t_max for more time resolution.
        is_impacting = tmp_is_impacting;
      } else {
        toi = tmp_is_impacting ? toi : t_max;
      }

      // This modification is for CCD-filtered line-search (e.g., IPC)
      // strategies for dealing with toi = 0:
      // 1. shrink t_max (when reaches max_itr),
      // 2. shrink tolerance (when not reach max_itr and tolerance is big) or
      // ms (when tolerance is too small comparing with ms)
      if (tmp_is_impacting && toi == 0 && no_zero_toi) {
        if (output_tolerance > tolerance) {
          // reaches max_itr, so shrink t_max to return a more accurate result to reach target
          // tolerance.
          t_max *= 0.9;
        } else if (10 * tolerance < ms) {
          ms *= 0.5;  // ms is too large, shrink it
        } else {
          tolerance *= 0.5;  // tolerance is too large, shrink it

          tol = compute_edge_edge_tolerances(a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e, tolerance);
        }
      }

      // Only perform a second iteration if toi == 0.
      // WARNING: This option assumes the initial distance is not zero.
    } while (no_zero_toi && ++no_zero_toi_iter < MAX_NO_ZERO_TOI_ITER && tmp_is_impacting
             && toi == 0);
    // assert(!no_zero_toi || !is_impacting || toi != 0);

    return is_impacting;
  }

  constexpr Array3 compute_face_vertex_tolerances(const Array3 &vs, const Array3 &f0s,
                                                  const Array3 &f1s, const Array3 &f2s,
                                                  const Array3 &ve, const Array3 &f0e,
                                                  const Array3 &f1e, const Array3 &f2e,
                                                  const double distance_tolerance) {
    const Array3 p000 = vs - f0s;
    const Array3 p001 = vs - f2s;
    const Array3 p011 = vs - (f1s + f2s - f0s);
    const Array3 p010 = vs - f1s;
    const Array3 p100 = ve - f0e;
    const Array3 p101 = ve - f2e;
    const Array3 p111 = ve - (f1e + f2e - f0e);
    const Array3 p110 = ve - f1e;

    double dl = 3 * max_linf_4(p000, p001, p011, p010, p100, p101, p111, p110);
    double edge0_length = 3 * max_linf_4(p000, p100, p101, p001, p010, p110, p111, p011);
    double edge1_length = 3 * max_linf_4(p000, p100, p110, p010, p001, p101, p111, p011);

    return Array3(std::min(distance_tolerance / dl, CCD_MAX_TIME_TOL),
                  std::min(distance_tolerance / edge0_length, CCD_MAX_COORD_TOL),
                  std::min(distance_tolerance / edge1_length, CCD_MAX_COORD_TOL));
  }
  constexpr bool vertexFaceCCD(const Array3 &vertex_start, const Array3 &face_vertex0_start,
                               const Array3 &face_vertex1_start, const Array3 &face_vertex2_start,
                               const Array3 &vertex_end, const Array3 &face_vertex0_end,
                               const Array3 &face_vertex1_end, const Array3 &face_vertex2_end,
                               const Array3 &err_in, const double ms_in, double &toi,
                               const double tolerance_in, const double t_max_in, const int max_itr,
                               double &output_tolerance, bool &earlyTerminate,
                               bool no_zero_toi = true) {
    const int MAX_NO_ZERO_TOI_ITER = std::numeric_limits<int>::max();
    // unsigned so can be larger than MAX_NO_ZERO_TOI_ITER
    unsigned int no_zero_toi_iter = 0;

    earlyTerminate = false;
    bool is_impacting{}, tmp_is_impacting{};

    // Mutable copies for no_zero_toi
    double t_max = t_max_in;
    double tolerance = tolerance_in;
    double ms = ms_in;

    Array3 tol = compute_face_vertex_tolerances(
        vertex_start, face_vertex0_start, face_vertex1_start, face_vertex2_start, vertex_end,
        face_vertex0_end, face_vertex1_end, face_vertex2_end, tolerance);

    //////////////////////////////////////////////////////////
    // this is the error of the whole mesh
    Array3 err{};
    // if error[0]<0, means we need to calculate error here
    if (err_in[0] < 0) {
      zs::vec<Array3, 8> vlist{vertex_start,       face_vertex0_start, face_vertex1_start,
                               face_vertex2_start, vertex_end,         face_vertex0_end,
                               face_vertex1_end,   face_vertex2_end};
      bool use_ms = ms > 0;
      err = get_numerical_error(vlist, true, use_ms);
    } else {
      err = err_in;
    }
    //////////////////////////////////////////////////////////

    do {
#if 1
      // no handling for zero toi
      return interval_root_finder_DFS<true>(vertex_start, face_vertex0_start, face_vertex1_start,
                                            face_vertex2_start, vertex_end, face_vertex0_end,
                                            face_vertex1_end, face_vertex2_end, tol, err, ms, toi);
      // assert(t_max >= 0 && t_max <= 1);
#else
      tmp_is_impacting = interval_root_finder_BFS<true>(
          vertex_start, face_vertex0_start, face_vertex1_start, face_vertex2_start, vertex_end,
          face_vertex0_end, face_vertex1_end, face_vertex2_end, tol, tolerance, err, ms, t_max,
          max_itr, toi, output_tolerance, earlyTerminate);
      // if (tmp_is_impacting) return true;
#endif
      // assert(!tmp_is_impacting || toi >= 0);

      if (t_max == t_max_in) {
        // This will be the final output because we might need to
        // perform CCD again if the toi is zero. In which case we will
        // use a smaller t_max for more time resolution.
        is_impacting = tmp_is_impacting;
      } else {
        toi = tmp_is_impacting ? toi : t_max;
      }

      // This modification is for CCD-filtered line-search (e.g., IPC)
      // strategies for dealing with toi = 0:
      // 1. shrink t_max (when reaches max_itr),
      // 2. shrink tolerance (when not reach max_itr and tolerance is big) or
      // ms (when tolerance is too small comparing with ms)
      if (tmp_is_impacting && toi == 0 && no_zero_toi) {
        if (output_tolerance > tolerance) {
          // reaches max_itr, so shrink t_max to return a more accurate result to reach target
          // tolerance.
          t_max *= 0.9;
        } else if (10 * tolerance < ms) {
          ms *= 0.5;  // ms is too large, shrink it
        } else {
          tolerance *= 0.5;  // tolerance is too large, shrink it

          // recompute this
          tol = compute_face_vertex_tolerances(vertex_start, face_vertex0_start, face_vertex1_start,
                                               face_vertex2_start, vertex_end, face_vertex0_end,
                                               face_vertex1_end, face_vertex2_end, tolerance);
        }
      }

      // Only perform a second iteration if toi == 0.
      // WARNING: This option assumes the initial distance is not zero.
    } while (no_zero_toi && ++no_zero_toi_iter < MAX_NO_ZERO_TOI_ITER && tmp_is_impacting
             && toi == 0);
    // assert(!no_zero_toi || !is_impacting || toi != 0);

    return is_impacting;
  }

}  // namespace zs