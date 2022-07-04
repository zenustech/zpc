/// reference: igl/predicates
// http://www.cs.cmu.edu/~quake/robust.html
// http://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c
/*****************************************************************************/
/*                                                                           */
/*  Routines for Arbitrary Precision Floating-point Arithmetic               */
/*  and Fast Robust Geometric Predicates                                     */
/*  (predicates.c)                                                           */
/*                                                                           */
/*  May 18, 1996                                                             */
/*                                                                           */
/*  Placed in the public domain by                                           */
/*  Jonathan Richard Shewchuk                                                */
/*  School of Computer Science                                               */
/*  Carnegie Mellon University                                               */
/*  5000 Forbes Avenue                                                       */
/*  Pittsburgh, Pennsylvania  15213-3891                                     */
/*  jrs@cs.cmu.edu                                                           */
/*                                                                           */
/*  This file contains C implementation of algorithms for exact addition     */
/*    and multiplication of floating-point numbers, and predicates for       */
/*    robustly performing the orientation and incircle tests used in         */
/*    computational geometry.  The algorithms and underlying theory are      */
/*    described in Jonathan Richard Shewchuk.  "Adaptive Precision Floating- */
/*    Point Arithmetic and Fast Robust Geometric Predicates."  Technical     */
/*    Report CMU-CS-96-140, School of Computer Science, Carnegie Mellon      */
/*    University, Pittsburgh, Pennsylvania, May 1996.  (Submitted to         */
/*    Discrete & Computational Geometry.)                                    */
/*                                                                           */
/*  This file, the paper listed above, and other information are available   */
/*    from the Web page http://www.cs.cmu.edu/~quake/robust.html .           */
/*                                                                           */
/*****************************************************************************/

/*****************************************************************************/
/*                                                                           */
/*  Using this code:                                                         */
/*                                                                           */
/*  First, read the short or long version of the paper (from the Web page    */
/*    above).                                                                */
/*                                                                           */
/*  Be sure to call exactinit() once, before calling any of the arithmetic   */
/*    functions or geometric predicates.  Also be sure to turn on the        */
/*    optimizer when compiling this file.                                    */
/*                                                                           */
/*                                                                           */
/*  Several geometric predicates are defined.  Their parameters are all      */
/*    points.  Each point is an array of two or three floating-point         */
/*    numbers.  The geometric predicates, described in the papers, are       */
/*                                                                           */
/*    orient2d(pa, pb, pc)                                                   */
/*    orient2dfast(pa, pb, pc)                                               */
/*    orient3d(pa, pb, pc, pd)                                               */
/*    orient3dfast(pa, pb, pc, pd)                                           */
/*    incircle(pa, pb, pc, pd)                                               */
/*    incirclefast(pa, pb, pc, pd)                                           */
/*    insphere(pa, pb, pc, pd, pe)                                           */
/*    inspherefast(pa, pb, pc, pd, pe)                                       */
/*                                                                           */
/*  Those with suffix "fast" are approximate, non-robust versions.  Those    */
/*    without the suffix are adaptive precision, robust versions.  There     */
/*    are also versions with the suffices "exact" and "slow", which are      */
/*    non-adaptive, exact arithmetic versions, which I use only for timings  */
/*    in my arithmetic papers.                                               */
/*                                                                           */
/*                                                                           */
/*  An expansion is represented by an array of floating-point numbers,       */
/*    sorted from smallest to largest magnitude (possibly with interspersed  */
/*    zeros).  The length of each expansion is stored as a separate integer, */
/*    and each arithmetic function returns an integer which is the length    */
/*    of the expansion it created.                                           */
/*                                                                           */
/*  Several arithmetic functions are defined.  Their parameters are          */
/*                                                                           */
/*    e, f           Input expansions                                        */
/*    elen, flen     Lengths of input expansions (must be >= 1)              */
/*    h              Output expansion                                        */
/*    b              Input scalar                                            */
/*                                                                           */
/*  The arithmetic functions are                                             */
/*                                                                           */
/*    grow_expansion(elen, e, b, h)                                          */
/*    grow_expansion_zeroelim(elen, e, b, h)                                 */
/*    expansion_sum(elen, e, flen, f, h)                                     */
/*    expansion_sum_zeroelim1(elen, e, flen, f, h)                           */
/*    expansion_sum_zeroelim2(elen, e, flen, f, h)                           */
/*    fast_expansion_sum(elen, e, flen, f, h)                                */
/*    fast_expansion_sum_zeroelim(elen, e, flen, f, h)                       */
/*    linear_expansion_sum(elen, e, flen, f, h)                              */
/*    linear_expansion_sum_zeroelim(elen, e, flen, f, h)                     */
/*    scale_expansion(elen, e, b, h)                                         */
/*    scale_expansion_zeroelim(elen, e, b, h)                                */
/*    compress(elen, e, h)                                                   */
/*                                                                           */
/*  All of these are described in the long version of the paper; some are    */
/*    described in the short version.  All return an integer that is the     */
/*    length of h.  Those with suffix _zeroelim perform zero elimination,    */
/*    and are recommended over their counterparts.  The procedure            */
/*    fast_expansion_sum_zeroelim() (or linear_expansion_sum_zeroelim() on   */
/*    processors that do not use the round-to-even tiebreaking rule) is      */
/*    recommended over expansion_sum_zeroelim().  Each procedure has a       */
/*    little note next to it (in the code below) that tells you whether or   */
/*    not the output expansion may be the same array as one of the input     */
/*    expansions.                                                            */
/*                                                                           */
/*                                                                           */
/*  If you look around below, you'll also find macros for a bunch of         */
/*    simple unrolled arithmetic operations, and procedures for printing     */
/*    expansions (commented out because they don't work with all C           */
/*    compilers) and for generating random floating-point numbers whose      */
/*    significand bits are all random.  Most of the macros have undocumented */
/*    requirements that certain of their parameters should not be the same   */
/*    variable; for safety, better to make sure all the parameters are       */
/*    distinct variables.  Feel free to send email to jrs@cs.cmu.edu if you  */
/*    have questions.                                                        */
/*                                                                           */
/*****************************************************************************/

#pragma once

namespace zs {

  template <int I> constexpr double init_exact() noexcept {
    double half = 0.5, check = 1.0, lastcheck{};
    int every_other = 1;
    double epsilon = 1.0;
    double splitter = 1.0;
    /* Repeatedly divide `epsilon' by two until it is too small to add to    */
    /*   one without causing roundoff.  (Also check if the sum is equal to   */
    /*   the previous sum, for machines that round up instead of using exact */
    /*   rounding.  Not that this library will work on such machines anyway. */
    do {
      lastcheck = check;
      epsilon *= half;
      if (every_other) {
        splitter *= 2.0;
      }
      every_other = !every_other;
      check = 1.0 + epsilon;
    } while ((check != 1.0) && (check != lastcheck));
    splitter += 1.0;
    /* Error bounds for orientation and incircle tests. */
    auto resulterrbound = (3.0 + 8.0 * epsilon) * epsilon;
    auto ccwerrboundA = (3.0 + 16.0 * epsilon) * epsilon;
    auto ccwerrboundB = (2.0 + 12.0 * epsilon) * epsilon;
    auto ccwerrboundC = (9.0 + 64.0 * epsilon) * epsilon * epsilon;
    auto o3derrboundA = (7.0 + 56.0 * epsilon) * epsilon;
    auto o3derrboundB = (3.0 + 28.0 * epsilon) * epsilon;
    auto o3derrboundC = (26.0 + 288.0 * epsilon) * epsilon * epsilon;
    auto iccerrboundA = (10.0 + 96.0 * epsilon) * epsilon;
    auto iccerrboundB = (4.0 + 48.0 * epsilon) * epsilon;
    auto iccerrboundC = (44.0 + 576.0 * epsilon) * epsilon * epsilon;
    auto isperrboundA = (16.0 + 224.0 * epsilon) * epsilon;
    auto isperrboundB = (5.0 + 72.0 * epsilon) * epsilon;
    auto isperrboundC = (71.0 + 1408.0 * epsilon) * epsilon * epsilon;
    if constexpr (I == 0)
      return epsilon;
    else if constexpr (I == 1)
      return splitter;
    else if constexpr (I == 2)
      return resulterrbound;
    // ccw
    else if constexpr (I == 3)
      return ccwerrboundA;
    else if constexpr (I == 4)
      return ccwerrboundB;
    else if constexpr (I == 5)
      return ccwerrboundC;
    // o3d
    else if constexpr (I == 6)
      return o3derrboundA;
    else if constexpr (I == 7)
      return o3derrboundB;
    else if constexpr (I == 8)
      return o3derrboundC;
    // icc
    else if constexpr (I == 9)
      return iccerrboundA;
    else if constexpr (I == 10)
      return iccerrboundB;
    else if constexpr (I == 11)
      return iccerrboundC;
    // isp
    else if constexpr (I == 12)
      return isperrboundA;
    else if constexpr (I == 13)
      return isperrboundB;
    else if constexpr (I == 14)
      return isperrboundC;
    return 0.0;
  }

  constexpr double g_epsilon = init_exact<0>();
  constexpr double g_splitter = init_exact<1>();
  constexpr double g_resulterrbound = init_exact<2>();
  constexpr double g_ccwerrboundA = init_exact<3>();
  constexpr double g_ccwerrboundB = init_exact<4>();
  constexpr double g_ccwerrboundC = init_exact<5>();
  constexpr double g_o3derrboundA = init_exact<6>();
  constexpr double g_o3derrboundB = init_exact<7>();
  constexpr double g_o3derrboundC = init_exact<8>();
  constexpr double g_iccerrboundA = init_exact<9>();
  constexpr double g_iccerrboundB = init_exact<10>();
  constexpr double g_iccerrboundC = init_exact<11>();
  constexpr double g_isperrboundA = init_exact<12>();
  constexpr double g_isperrboundB = init_exact<13>();
  constexpr double g_isperrboundC = init_exact<14>();

  constexpr double absolute(double a) noexcept { return a >= 0.0 ? a : -a; }
  // two diff
  constexpr void two_diff_tail(double a, double b, double x, double &y) noexcept {
    auto bvirt = a - x;
    auto avirt = x + bvirt;
    auto bround = bvirt - b;
    auto around = a - avirt;
    y = around + bround;
  }
  constexpr void two_diff(double a, double b, double &x, double &y) noexcept {
    x = a - b;
    two_diff_tail(a, b, x, y);
  }

  // two sum
  constexpr void two_sum_tail(double a, double b, double x, double &y) noexcept {
    auto bvirt = x - a;
    auto avirt = x - bvirt;
    auto bround = b - bvirt;
    auto around = a - avirt;
    y = around + bround;
  }
  constexpr void two_sum(double a, double b, double &x, double &y) noexcept {
    x = a + b;
    two_sum_tail(a, b, x, y);
  }
  constexpr void fast_two_sum_tail(double a, double b, double x, double &y) noexcept {
    auto bvirt = x - a;
    y = b - bvirt;
  }
  constexpr void fast_two_sum(double a, double b, double &x, double &y) noexcept {
    x = a + b;
    fast_two_sum_tail(a, b, x, y);
  }

  // two product
  constexpr void split(double a, double &ahi, double &alo) noexcept {
    auto c = g_splitter * a;
    auto abig = c - a;
    ahi = c - abig;
    alo = a - ahi;
  }
  constexpr void two_product_tail(double a, double b, double x, double &y) noexcept {
    double ahi{}, alo{};
    split(a, ahi, alo);
    double bhi{}, blo{};
    split(b, bhi, blo);
    auto err1 = x - (ahi * bhi);
    auto err2 = err1 - (alo * bhi);
    auto err3 = err2 - (ahi * blo);
    y = alo * blo - err3;
  }
  constexpr void two_product(double a, double b, double &x, double &y) noexcept {
    x = a * b;
    two_product_tail(a, b, x, y);
  }

  // two one
  constexpr void two_one_sum(double a1, double a0, double b, double &x2, double &x1,
                             double &x0) noexcept {
    double _i{};
    two_sum(a0, b, _i, x0);
    two_sum(a1, _i, x2, x1);
  }
  constexpr void two_one_diff(double a1, double a0, double b, double &x2, double &x1,
                              double &x0) noexcept {
    double _i{};
    two_diff(a0, b, _i, x0);
    two_sum(a1, _i, x2, x1);
  }
  /* two_product_presplit() is Two_Product() where one of the inputs has       */
  /*   already been split.  Avoids redundant splitting.                        */
  constexpr void two_product_presplit(double a, double b, double bhi, double blo, double &x,
                                      double &y) noexcept {
    x = a * b;
    double ahi{}, alo{};
    split(a, ahi, alo);
    auto err1 = x - (ahi * bhi);
    auto err2 = err1 - (alo * bhi);
    auto err3 = err2 - (ahi * blo);
    y = (alo * blo) - err3;
  }
  constexpr void two_one_product(double a1, double a0, double b, double &x3, double &x2, double &x1,
                                 double &x0) noexcept {
    double bhi{}, blo{};
    split(b, bhi, blo);
    double _i{}, _j{}, _0{}, _k{};
    two_product_presplit(a0, b, bhi, blo, _i, x0);
    two_product_presplit(a1, b, bhi, blo, _j, _0);
    two_sum(_i, _0, _k, x1);
    fast_two_sum(_j, _k, x3, x2);
  }

  // two two
  constexpr void two_two_sum(double a1, double a0, double b1, double b0, double &x3, double &x2,
                             double &x1, double &x0) noexcept {
    double _j{}, _0{};
    two_one_sum(a1, a0, b0, _j, _0, x0);
    two_one_sum(_j, _0, b1, x3, x2, x1);
  }
  constexpr void two_two_diff(double a1, double a0, double b1, double b0, double &x3, double &x2,
                              double &x1, double &x0) noexcept {
    double _j{}, _0{};
    two_one_diff(a1, a0, b0, _j, _0, x0);
    two_one_diff(_j, _0, b1, x3, x2, x1);
  }

  /*****************************************************************************/
  /*                                                                           */
  /*  estimate()   Produce a one-word estimate of an expansion's value.        */
  /*                                                                           */
  /*  See either version of my paper for details.                              */
  /*                                                                           */
  /*****************************************************************************/

  constexpr double estimate(int elen, const double *e) noexcept {
    double Q{};
    int eindex{};

    Q = e[0];
    for (eindex = 1; eindex < elen; eindex++) {
      Q += e[eindex];
    }
    return Q;
  }

  /*****************************************************************************/
  /*                                                                           */
  /*  scale_expansion_zeroelim()   Multiply an expansion by a scalar,          */
  /*                               eliminating zero components from the        */
  /*                               output expansion.                           */
  /*                                                                           */
  /*  Sets h = be.  See either version of my paper for details.                */
  /*                                                                           */
  /*  Maintains the nonoverlapping property.  If round-to-even is used (as     */
  /*  with IEEE 754), maintains the strongly nonoverlapping and nonadjacent    */
  /*  properties as well.  (That is, if e has one of these properties, so      */
  /*  will h.)                                                                 */
  /*                                                                           */
  /*****************************************************************************/

  constexpr int scale_expansion_zeroelim(int elen, const double *e, double b,
                                         double *h) /* e and h cannot be the same. */
      noexcept {
    double Q{}, sum{};
    double hh{};
    double product1{};
    double product0{};
    int eindex{}, hindex{};
    double enow{};
    double bhi{}, blo{};

    split(b, bhi, blo);
    two_product_presplit(e[0], b, bhi, blo, Q, hh);
    hindex = 0;
    if (hh != 0) {
      h[hindex++] = hh;
    }
    for (eindex = 1; eindex < elen; eindex++) {
      enow = e[eindex];
      two_product_presplit(enow, b, bhi, blo, product1, product0);
      two_sum(Q, product0, sum, hh);
      if (hh != 0) {
        h[hindex++] = hh;
      }
      fast_two_sum(product1, sum, Q, hh);
      if (hh != 0) {
        h[hindex++] = hh;
      }
    }
    if ((Q != 0.0) || (hindex == 0)) {
      h[hindex++] = Q;
    }
    return hindex;
  }

  /*****************************************************************************/
  /*                                                                           */
  /*  fast_expansion_sum_zeroelim()   Sum two expansions, eliminating zero     */
  /*                                  components from the output expansion.    */
  /*                                                                           */
  /*  Sets h = e + f.  See the long version of my paper for details.           */
  /*                                                                           */
  /*  If round-to-even is used (as with IEEE 754), maintains the strongly      */
  /*  nonoverlapping property.  (That is, if e is strongly nonoverlapping, h   */
  /*  will be also.)  Does NOT maintain the nonoverlapping or nonadjacent      */
  /*  properties.                                                              */
  /*                                                                           */
  /*****************************************************************************/

  constexpr int fast_expansion_sum_zeroelim(int elen, const double *e, int flen, const double *f,
                                            double *h) /* h cannot be e or f. */
      noexcept {
    double Q{};
    double Qnew{};
    double hh{};
    int eindex{}, findex{}, hindex{};
    double enow{}, fnow{};

    enow = e[0];
    fnow = f[0];
    eindex = findex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
      Q = enow;
      enow = e[++eindex];
    } else {
      Q = fnow;
      fnow = f[++findex];
    }
    hindex = 0;
    if ((eindex < elen) && (findex < flen)) {
      if ((fnow > enow) == (fnow > -enow)) {
        fast_two_sum(enow, Q, Qnew, hh);
        enow = e[++eindex];
      } else {
        fast_two_sum(fnow, Q, Qnew, hh);
        fnow = f[++findex];
      }
      Q = Qnew;
      if (hh != 0.0) {
        h[hindex++] = hh;
      }
      while ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
          two_sum(Q, enow, Qnew, hh);
          enow = e[++eindex];
        } else {
          two_sum(Q, fnow, Qnew, hh);
          fnow = f[++findex];
        }
        Q = Qnew;
        if (hh != 0.0) {
          h[hindex++] = hh;
        }
      }
    }
    while (eindex < elen) {
      two_sum(Q, enow, Qnew, hh);
      enow = e[++eindex];
      Q = Qnew;
      if (hh != 0.0) {
        h[hindex++] = hh;
      }
    }
    while (findex < flen) {
      two_sum(Q, fnow, Qnew, hh);
      fnow = f[++findex];
      Q = Qnew;
      if (hh != 0.0) {
        h[hindex++] = hh;
      }
    }
    if ((Q != 0.0) || (hindex == 0)) {
      h[hindex++] = Q;
    }
    return hindex;
  }

  /*****************************************************************************/
  /*                                                                           */
  /*  orient2dfast()   Approximate 2D orientation test.  Nonrobust.            */
  /*  orient2dexact()   Exact 2D orientation test.  Robust.                    */
  /*  orient2dslow()   Another exact 2D orientation test.  Robust.             */
  /*  orient2d()   Adaptive exact 2D orientation test.  Robust.                */
  /*                                                                           */
  /*               Return a positive value if the points pa, pb, and pc occur  */
  /*               in counterclockwise order; a negative value if they occur   */
  /*               in clockwise order; and zero if they are collinear.  The    */
  /*               result is also a rough approximation of twice the signed    */
  /*               area of the triangle defined by the three points.           */
  /*                                                                           */
  /*  Only the first and last routine should be used; the middle two are for   */
  /*  timings.                                                                 */
  /*                                                                           */
  /*  The last three use exact arithmetic to ensure a correct answer.  The     */
  /*  result returned is the determinant of a matrix.  In orient2d() only,     */
  /*  this determinant is computed adaptively, in the sense that exact         */
  /*  arithmetic is used only to the degree it is needed to ensure that the    */
  /*  returned value has the correct sign.  Hence, orient2d() is usually quite */
  /*  fast, but will run more slowly when the input points are collinear or    */
  /*  nearly so.                                                               */
  /*                                                                           */
  /*****************************************************************************/
  constexpr double orient2dadapt(const double *pa, const double *pb, const double *pc,
                                 double detsum) noexcept {
    double acx{}, acy{}, bcx{}, bcy{};
    double acxtail{}, acytail{}, bcxtail{}, bcytail{};
    double detleft{}, detright{};
    double detlefttail{}, detrighttail{};
    double det{}, errbound{};
    double B[4] = {}, C1[8] = {}, C2[12] = {}, D[16] = {};
    double B3{};
    int C1length{}, C2length{}, Dlength{};
    double u[4] = {};
    double u3{};
    double s1{}, t1{};
    double s0{}, t0{};

    acx = (double)(pa[0] - pc[0]);
    bcx = (double)(pb[0] - pc[0]);
    acy = (double)(pa[1] - pc[1]);
    bcy = (double)(pb[1] - pc[1]);

    two_product(acx, bcy, detleft, detlefttail);
    two_product(acy, bcx, detright, detrighttail);

    two_two_diff(detleft, detlefttail, detright, detrighttail, B3, B[2], B[1], B[0]);
    B[3] = B3;

    det = estimate(4, B);
    errbound = g_ccwerrboundB * detsum;
    if ((det >= errbound) || (-det >= errbound)) {
      return det;
    }

    two_diff_tail(pa[0], pc[0], acx, acxtail);
    two_diff_tail(pb[0], pc[0], bcx, bcxtail);
    two_diff_tail(pa[1], pc[1], acy, acytail);
    two_diff_tail(pb[1], pc[1], bcy, bcytail);

    if ((acxtail == 0.0) && (acytail == 0.0) && (bcxtail == 0.0) && (bcytail == 0.0)) {
      return det;
    }

    errbound = g_ccwerrboundC * detsum + g_resulterrbound * absolute(det);
    det += (acx * bcytail + bcy * acxtail) - (acy * bcxtail + bcx * acytail);
    if ((det >= errbound) || (-det >= errbound)) {
      return det;
    }

    two_product(acxtail, bcy, s1, s0);
    two_product(acytail, bcx, t1, t0);
    two_two_diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    C1length = fast_expansion_sum_zeroelim(4, B, 4, u, C1);

    two_product(acx, bcytail, s1, s0);
    two_product(acy, bcxtail, t1, t0);
    two_two_diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    C2length = fast_expansion_sum_zeroelim(C1length, C1, 4, u, C2);

    two_product(acxtail, bcytail, s1, s0);
    two_product(acytail, bcxtail, t1, t0);
    two_two_diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    Dlength = fast_expansion_sum_zeroelim(C2length, C2, 4, u, D);

    return (D[Dlength - 1]);
  }

  constexpr double orient2d(const double *pa, const double *pb, const double *pc) noexcept {
    double detleft{}, detright{}, det{};
    double detsum{}, errbound{};

    detleft = (pa[0] - pc[0]) * (pb[1] - pc[1]);
    detright = (pa[1] - pc[1]) * (pb[0] - pc[0]);
    det = detleft - detright;

    if (detleft > 0.0) {
      if (detright <= 0.0) {
        return det;
      } else {
        detsum = detleft + detright;
      }
    } else if (detleft < 0.0) {
      if (detright >= 0.0) {
        return det;
      } else {
        detsum = -detleft - detright;
      }
    } else {
      return det;
    }

    errbound = g_ccwerrboundA * detsum;
    if ((det >= errbound) || (-det >= errbound)) {
      return det;
    }

    return orient2dadapt(pa, pb, pc, detsum);
  }

  /*****************************************************************************/
  /*                                                                           */
  /*  orient3dfast()   Approximate 3D orientation test.  Nonrobust.            */
  /*  orient3dexact()   Exact 3D orientation test.  Robust.                    */
  /*  orient3dslow()   Another exact 3D orientation test.  Robust.             */
  /*  orient3d()   Adaptive exact 3D orientation test.  Robust.                */
  /*                                                                           */
  /*               Return a positive value if the point pd lies below the      */
  /*               plane passing through pa, pb, and pc; "below" is defined so */
  /*               that pa, pb, and pc appear in counterclockwise order when   */
  /*               viewed from above the plane.  Returns a negative value if   */
  /*               pd lies above the plane.  Returns zero if the points are    */
  /*               coplanar.  The result is also a rough approximation of six  */
  /*               times the signed volume of the tetrahedron defined by the   */
  /*               four points.                                                */
  /*                                                                           */
  /*  Only the first and last routine should be used; the middle two are for   */
  /*  timings.                                                                 */
  /*                                                                           */
  /*  The last three use exact arithmetic to ensure a correct answer.  The     */
  /*  result returned is the determinant of a matrix.  In orient3d() only,     */
  /*  this determinant is computed adaptively, in the sense that exact         */
  /*  arithmetic is used only to the degree it is needed to ensure that the    */
  /*  returned value has the correct sign.  Hence, orient3d() is usually quite */
  /*  fast, but will run more slowly when the input points are coplanar or     */
  /*  nearly so.                                                               */
  /*                                                                           */
  /*****************************************************************************/
  constexpr double orient3dadapt(const double *pa, const double *pb, const double *pc,
                                 const double *pd, double permanent) noexcept {
    double adx{}, bdx{}, cdx{}, ady{}, bdy{}, cdy{}, adz{}, bdz{}, cdz{};
    double det{}, errbound{};

    double bdxcdy1{}, cdxbdy1{}, cdxady1{}, adxcdy1{}, adxbdy1{}, bdxady1{};
    double bdxcdy0{}, cdxbdy0{}, cdxady0{}, adxcdy0{}, adxbdy0{}, bdxady0{};
    double bc[4] = {}, ca[4] = {}, ab[4] = {};
    double bc3{}, ca3{}, ab3{};
    double adet[8] = {}, bdet[8] = {}, cdet[8] = {};
    int alen{}, blen{}, clen{};
    double abdet[16] = {};
    int ablen{};
    double *finnow{nullptr}, *finother{nullptr}, *finswap{nullptr};
    double fin1[192] = {}, fin2[192] = {};
    int finlength{};

    double adxtail{}, bdxtail{}, cdxtail{};
    double adytail{}, bdytail{}, cdytail{};
    double adztail{}, bdztail{}, cdztail{};
    double at_blarge{}, at_clarge{};
    double bt_clarge{}, bt_alarge{};
    double ct_alarge{}, ct_blarge{};
    double at_b[4] = {}, at_c[4] = {}, bt_c[4] = {}, bt_a[4] = {}, ct_a[4] = {}, ct_b[4] = {};
    int at_blen{}, at_clen{}, bt_clen{}, bt_alen{}, ct_alen{}, ct_blen{};
    double bdxt_cdy1{}, cdxt_bdy1{}, cdxt_ady1{};
    double adxt_cdy1{}, adxt_bdy1{}, bdxt_ady1{};
    double bdxt_cdy0{}, cdxt_bdy0{}, cdxt_ady0{};
    double adxt_cdy0{}, adxt_bdy0{}, bdxt_ady0{};
    double bdyt_cdx1{}, cdyt_bdx1{}, cdyt_adx1{};
    double adyt_cdx1{}, adyt_bdx1{}, bdyt_adx1{};
    double bdyt_cdx0{}, cdyt_bdx0{}, cdyt_adx0{};
    double adyt_cdx0{}, adyt_bdx0{}, bdyt_adx0{};
    double bct[8] = {}, cat[8] = {}, abt[8] = {};
    int bctlen{}, catlen{}, abtlen{};
    double bdxt_cdyt1{}, cdxt_bdyt1{}, cdxt_adyt1{};
    double adxt_cdyt1{}, adxt_bdyt1{}, bdxt_adyt1{};
    double bdxt_cdyt0{}, cdxt_bdyt0{}, cdxt_adyt0{};
    double adxt_cdyt0{}, adxt_bdyt0{}, bdxt_adyt0{};
    double u[4] = {}, v[12] = {}, w[16] = {};
    double u3{};
    int vlength{}, wlength{};
    double negate{};

    adx = (double)(pa[0] - pd[0]);
    bdx = (double)(pb[0] - pd[0]);
    cdx = (double)(pc[0] - pd[0]);
    ady = (double)(pa[1] - pd[1]);
    bdy = (double)(pb[1] - pd[1]);
    cdy = (double)(pc[1] - pd[1]);
    adz = (double)(pa[2] - pd[2]);
    bdz = (double)(pb[2] - pd[2]);
    cdz = (double)(pc[2] - pd[2]);

    two_product(bdx, cdy, bdxcdy1, bdxcdy0);
    two_product(cdx, bdy, cdxbdy1, cdxbdy0);
    two_two_diff(bdxcdy1, bdxcdy0, cdxbdy1, cdxbdy0, bc3, bc[2], bc[1], bc[0]);
    bc[3] = bc3;
    alen = scale_expansion_zeroelim(4, bc, adz, adet);

    two_product(cdx, ady, cdxady1, cdxady0);
    two_product(adx, cdy, adxcdy1, adxcdy0);
    two_two_diff(cdxady1, cdxady0, adxcdy1, adxcdy0, ca3, ca[2], ca[1], ca[0]);
    ca[3] = ca3;
    blen = scale_expansion_zeroelim(4, ca, bdz, bdet);

    two_product(adx, bdy, adxbdy1, adxbdy0);
    two_product(bdx, ady, bdxady1, bdxady0);
    two_two_diff(adxbdy1, adxbdy0, bdxady1, bdxady0, ab3, ab[2], ab[1], ab[0]);
    ab[3] = ab3;
    clen = scale_expansion_zeroelim(4, ab, cdz, cdet);

    ablen = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
    finlength = fast_expansion_sum_zeroelim(ablen, abdet, clen, cdet, fin1);

    det = estimate(finlength, fin1);
    errbound = g_o3derrboundB * permanent;
    if ((det >= errbound) || (-det >= errbound)) {
      return det;
    }

    two_diff_tail(pa[0], pd[0], adx, adxtail);
    two_diff_tail(pb[0], pd[0], bdx, bdxtail);
    two_diff_tail(pc[0], pd[0], cdx, cdxtail);
    two_diff_tail(pa[1], pd[1], ady, adytail);
    two_diff_tail(pb[1], pd[1], bdy, bdytail);
    two_diff_tail(pc[1], pd[1], cdy, cdytail);
    two_diff_tail(pa[2], pd[2], adz, adztail);
    two_diff_tail(pb[2], pd[2], bdz, bdztail);
    two_diff_tail(pc[2], pd[2], cdz, cdztail);

    if ((adxtail == 0.0) && (bdxtail == 0.0) && (cdxtail == 0.0) && (adytail == 0.0)
        && (bdytail == 0.0) && (cdytail == 0.0) && (adztail == 0.0) && (bdztail == 0.0)
        && (cdztail == 0.0)) {
      return det;
    }

    errbound = g_o3derrboundC * permanent + g_resulterrbound * absolute(det);
    det += (adz * ((bdx * cdytail + cdy * bdxtail) - (bdy * cdxtail + cdx * bdytail))
            + adztail * (bdx * cdy - bdy * cdx))
           + (bdz * ((cdx * adytail + ady * cdxtail) - (cdy * adxtail + adx * cdytail))
              + bdztail * (cdx * ady - cdy * adx))
           + (cdz * ((adx * bdytail + bdy * adxtail) - (ady * bdxtail + bdx * adytail))
              + cdztail * (adx * bdy - ady * bdx));
    if ((det >= errbound) || (-det >= errbound)) {
      return det;
    }

    finnow = fin1;
    finother = fin2;

    if (adxtail == 0.0) {
      if (adytail == 0.0) {
        at_b[0] = 0.0;
        at_blen = 1;
        at_c[0] = 0.0;
        at_clen = 1;
      } else {
        negate = -adytail;
        two_product(negate, bdx, at_blarge, at_b[0]);
        at_b[1] = at_blarge;
        at_blen = 2;
        two_product(adytail, cdx, at_clarge, at_c[0]);
        at_c[1] = at_clarge;
        at_clen = 2;
      }
    } else {
      if (adytail == 0.0) {
        two_product(adxtail, bdy, at_blarge, at_b[0]);
        at_b[1] = at_blarge;
        at_blen = 2;
        negate = -adxtail;
        two_product(negate, cdy, at_clarge, at_c[0]);
        at_c[1] = at_clarge;
        at_clen = 2;
      } else {
        two_product(adxtail, bdy, adxt_bdy1, adxt_bdy0);
        two_product(adytail, bdx, adyt_bdx1, adyt_bdx0);
        two_two_diff(adxt_bdy1, adxt_bdy0, adyt_bdx1, adyt_bdx0, at_blarge, at_b[2], at_b[1],
                     at_b[0]);
        at_b[3] = at_blarge;
        at_blen = 4;
        two_product(adytail, cdx, adyt_cdx1, adyt_cdx0);
        two_product(adxtail, cdy, adxt_cdy1, adxt_cdy0);
        two_two_diff(adyt_cdx1, adyt_cdx0, adxt_cdy1, adxt_cdy0, at_clarge, at_c[2], at_c[1],
                     at_c[0]);
        at_c[3] = at_clarge;
        at_clen = 4;
      }
    }
    if (bdxtail == 0.0) {
      if (bdytail == 0.0) {
        bt_c[0] = 0.0;
        bt_clen = 1;
        bt_a[0] = 0.0;
        bt_alen = 1;
      } else {
        negate = -bdytail;
        two_product(negate, cdx, bt_clarge, bt_c[0]);
        bt_c[1] = bt_clarge;
        bt_clen = 2;
        two_product(bdytail, adx, bt_alarge, bt_a[0]);
        bt_a[1] = bt_alarge;
        bt_alen = 2;
      }
    } else {
      if (bdytail == 0.0) {
        two_product(bdxtail, cdy, bt_clarge, bt_c[0]);
        bt_c[1] = bt_clarge;
        bt_clen = 2;
        negate = -bdxtail;
        two_product(negate, ady, bt_alarge, bt_a[0]);
        bt_a[1] = bt_alarge;
        bt_alen = 2;
      } else {
        two_product(bdxtail, cdy, bdxt_cdy1, bdxt_cdy0);
        two_product(bdytail, cdx, bdyt_cdx1, bdyt_cdx0);
        two_two_diff(bdxt_cdy1, bdxt_cdy0, bdyt_cdx1, bdyt_cdx0, bt_clarge, bt_c[2], bt_c[1],
                     bt_c[0]);
        bt_c[3] = bt_clarge;
        bt_clen = 4;
        two_product(bdytail, adx, bdyt_adx1, bdyt_adx0);
        two_product(bdxtail, ady, bdxt_ady1, bdxt_ady0);
        two_two_diff(bdyt_adx1, bdyt_adx0, bdxt_ady1, bdxt_ady0, bt_alarge, bt_a[2], bt_a[1],
                     bt_a[0]);
        bt_a[3] = bt_alarge;
        bt_alen = 4;
      }
    }
    if (cdxtail == 0.0) {
      if (cdytail == 0.0) {
        ct_a[0] = 0.0;
        ct_alen = 1;
        ct_b[0] = 0.0;
        ct_blen = 1;
      } else {
        negate = -cdytail;
        two_product(negate, adx, ct_alarge, ct_a[0]);
        ct_a[1] = ct_alarge;
        ct_alen = 2;
        two_product(cdytail, bdx, ct_blarge, ct_b[0]);
        ct_b[1] = ct_blarge;
        ct_blen = 2;
      }
    } else {
      if (cdytail == 0.0) {
        two_product(cdxtail, ady, ct_alarge, ct_a[0]);
        ct_a[1] = ct_alarge;
        ct_alen = 2;
        negate = -cdxtail;
        two_product(negate, bdy, ct_blarge, ct_b[0]);
        ct_b[1] = ct_blarge;
        ct_blen = 2;
      } else {
        two_product(cdxtail, ady, cdxt_ady1, cdxt_ady0);
        two_product(cdytail, adx, cdyt_adx1, cdyt_adx0);
        two_two_diff(cdxt_ady1, cdxt_ady0, cdyt_adx1, cdyt_adx0, ct_alarge, ct_a[2], ct_a[1],
                     ct_a[0]);
        ct_a[3] = ct_alarge;
        ct_alen = 4;
        two_product(cdytail, bdx, cdyt_bdx1, cdyt_bdx0);
        two_product(cdxtail, bdy, cdxt_bdy1, cdxt_bdy0);
        two_two_diff(cdyt_bdx1, cdyt_bdx0, cdxt_bdy1, cdxt_bdy0, ct_blarge, ct_b[2], ct_b[1],
                     ct_b[0]);
        ct_b[3] = ct_blarge;
        ct_blen = 4;
      }
    }

    bctlen = fast_expansion_sum_zeroelim(bt_clen, bt_c, ct_blen, ct_b, bct);
    wlength = scale_expansion_zeroelim(bctlen, bct, adz, w);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w, finother);
    finswap = finnow;
    finnow = finother;
    finother = finswap;

    catlen = fast_expansion_sum_zeroelim(ct_alen, ct_a, at_clen, at_c, cat);
    wlength = scale_expansion_zeroelim(catlen, cat, bdz, w);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w, finother);
    finswap = finnow;
    finnow = finother;
    finother = finswap;

    abtlen = fast_expansion_sum_zeroelim(at_blen, at_b, bt_alen, bt_a, abt);
    wlength = scale_expansion_zeroelim(abtlen, abt, cdz, w);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w, finother);
    finswap = finnow;
    finnow = finother;
    finother = finswap;

    if (adztail != 0.0) {
      vlength = scale_expansion_zeroelim(4, bc, adztail, v);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, vlength, v, finother);
      finswap = finnow;
      finnow = finother;
      finother = finswap;
    }
    if (bdztail != 0.0) {
      vlength = scale_expansion_zeroelim(4, ca, bdztail, v);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, vlength, v, finother);
      finswap = finnow;
      finnow = finother;
      finother = finswap;
    }
    if (cdztail != 0.0) {
      vlength = scale_expansion_zeroelim(4, ab, cdztail, v);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, vlength, v, finother);
      finswap = finnow;
      finnow = finother;
      finother = finswap;
    }

    if (adxtail != 0.0) {
      if (bdytail != 0.0) {
        two_product(adxtail, bdytail, adxt_bdyt1, adxt_bdyt0);
        two_one_product(adxt_bdyt1, adxt_bdyt0, cdz, u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u, finother);
        finswap = finnow;
        finnow = finother;
        finother = finswap;
        if (cdztail != 0.0) {
          two_one_product(adxt_bdyt1, adxt_bdyt0, cdztail, u3, u[2], u[1], u[0]);
          u[3] = u3;
          finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u, finother);
          finswap = finnow;
          finnow = finother;
          finother = finswap;
        }
      }
      if (cdytail != 0.0) {
        negate = -adxtail;
        two_product(negate, cdytail, adxt_cdyt1, adxt_cdyt0);
        two_one_product(adxt_cdyt1, adxt_cdyt0, bdz, u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u, finother);
        finswap = finnow;
        finnow = finother;
        finother = finswap;
        if (bdztail != 0.0) {
          two_one_product(adxt_cdyt1, adxt_cdyt0, bdztail, u3, u[2], u[1], u[0]);
          u[3] = u3;
          finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u, finother);
          finswap = finnow;
          finnow = finother;
          finother = finswap;
        }
      }
    }
    if (bdxtail != 0.0) {
      if (cdytail != 0.0) {
        two_product(bdxtail, cdytail, bdxt_cdyt1, bdxt_cdyt0);
        two_one_product(bdxt_cdyt1, bdxt_cdyt0, adz, u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u, finother);
        finswap = finnow;
        finnow = finother;
        finother = finswap;
        if (adztail != 0.0) {
          two_one_product(bdxt_cdyt1, bdxt_cdyt0, adztail, u3, u[2], u[1], u[0]);
          u[3] = u3;
          finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u, finother);
          finswap = finnow;
          finnow = finother;
          finother = finswap;
        }
      }
      if (adytail != 0.0) {
        negate = -bdxtail;
        two_product(negate, adytail, bdxt_adyt1, bdxt_adyt0);
        two_one_product(bdxt_adyt1, bdxt_adyt0, cdz, u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u, finother);
        finswap = finnow;
        finnow = finother;
        finother = finswap;
        if (cdztail != 0.0) {
          two_one_product(bdxt_adyt1, bdxt_adyt0, cdztail, u3, u[2], u[1], u[0]);
          u[3] = u3;
          finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u, finother);
          finswap = finnow;
          finnow = finother;
          finother = finswap;
        }
      }
    }
    if (cdxtail != 0.0) {
      if (adytail != 0.0) {
        two_product(cdxtail, adytail, cdxt_adyt1, cdxt_adyt0);
        two_one_product(cdxt_adyt1, cdxt_adyt0, bdz, u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u, finother);
        finswap = finnow;
        finnow = finother;
        finother = finswap;
        if (bdztail != 0.0) {
          two_one_product(cdxt_adyt1, cdxt_adyt0, bdztail, u3, u[2], u[1], u[0]);
          u[3] = u3;
          finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u, finother);
          finswap = finnow;
          finnow = finother;
          finother = finswap;
        }
      }
      if (bdytail != 0.0) {
        negate = -cdxtail;
        two_product(negate, bdytail, cdxt_bdyt1, cdxt_bdyt0);
        two_one_product(cdxt_bdyt1, cdxt_bdyt0, adz, u3, u[2], u[1], u[0]);
        u[3] = u3;
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u, finother);
        finswap = finnow;
        finnow = finother;
        finother = finswap;
        if (adztail != 0.0) {
          two_one_product(cdxt_bdyt1, cdxt_bdyt0, adztail, u3, u[2], u[1], u[0]);
          u[3] = u3;
          finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u, finother);
          finswap = finnow;
          finnow = finother;
          finother = finswap;
        }
      }
    }

    if (adztail != 0.0) {
      wlength = scale_expansion_zeroelim(bctlen, bct, adztail, w);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w, finother);
      finswap = finnow;
      finnow = finother;
      finother = finswap;
    }
    if (bdztail != 0.0) {
      wlength = scale_expansion_zeroelim(catlen, cat, bdztail, w);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w, finother);
      finswap = finnow;
      finnow = finother;
      finother = finswap;
    }
    if (cdztail != 0.0) {
      wlength = scale_expansion_zeroelim(abtlen, abt, cdztail, w);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w, finother);
      finswap = finnow;
      finnow = finother;
      finother = finswap;
    }

    return finnow[finlength - 1];
  }

  constexpr double orient3d(const double *pa, const double *pb, const double *pc,
                            const double *pd) noexcept {
    double adx{}, bdx{}, cdx{}, ady{}, bdy{}, cdy{}, adz{}, bdz{}, cdz{};
    double bdxcdy{}, cdxbdy{}, cdxady{}, adxcdy{}, adxbdy{}, bdxady{};
    double det{};
    double permanent{}, errbound{};

    adx = pa[0] - pd[0];
    bdx = pb[0] - pd[0];
    cdx = pc[0] - pd[0];
    ady = pa[1] - pd[1];
    bdy = pb[1] - pd[1];
    cdy = pc[1] - pd[1];
    adz = pa[2] - pd[2];
    bdz = pb[2] - pd[2];
    cdz = pc[2] - pd[2];

    bdxcdy = bdx * cdy;
    cdxbdy = cdx * bdy;

    cdxady = cdx * ady;
    adxcdy = adx * cdy;

    adxbdy = adx * bdy;
    bdxady = bdx * ady;

    det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);

    permanent = (absolute(bdxcdy) + absolute(cdxbdy)) * absolute(adz)
                + (absolute(cdxady) + absolute(adxcdy)) * absolute(bdz)
                + (absolute(adxbdy) + absolute(bdxady)) * absolute(cdz);
    errbound = g_o3derrboundA * permanent;
    if ((det > errbound) || (-det > errbound)) {
      return det;
    }

    return orient3dadapt(pa, pb, pc, pd, permanent);
  }

}  // namespace zs