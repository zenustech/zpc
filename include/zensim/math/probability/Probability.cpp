#include "Probability.h"

#include <cmath>

#include "zensim/math/MathUtils.h"
#include "zensim/math/Rotation.hpp"

namespace zs {

  double PDF(int lambda, int k) {
    double pdf = 1;
    for (int i = 1; i <= k; ++i) pdf *= (double)lambda / i;
    return pdf * zs::exp(-1.0 * lambda);
  }

  double PDF(double u, double o, int x) {
    const double co = 1. / zs::sqrt(2. * g_pi);
    double index = -(x - u) * (x - u) / 2. / o / o;
    return co / o * zs::exp(index);
  }
  double anti_normal_PDF(double u, double o, int x) {
    const double co = 1. / zs::sqrt(2. * g_pi);
    double index = -(x - u) * (x - u) / 2. / o / o;
    return 1 - co / o * zs::exp(index);
  }

  int rand_p(double lambda) {
    double u = (double)rand() / RAND_MAX;
    int x = 0;
    double cdf = zs::exp(-1.0 * lambda);
    while (u >= cdf) {
      x++;
      cdf += PDF(lambda, x);
    }
    return x;
  }
  int rand_normal(double u, double o) {
    double val = (double)rand() / RAND_MAX;
    int x = 0;
    double cdf = 0;  // PDF(u, o, x)
    while (val >= cdf) {
      x++;
      cdf += PDF(u, o, x);
    }
    return x;
  }
  int rand_anti_normal(double u, double o) {
    double val = (double)rand() / RAND_MAX;
    int x = 0;
    double cdf = 0;  // PDF(u, o, x)
    while (val >= cdf) {
      x++;
      cdf += anti_normal_PDF(u, o, x);
    }
    return x;
  }
}  // namespace zs