#pragma once
#include "zensim/container/TileVector.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/math/probability/Random.hpp"
#if ZS_ENABLE_OPENMP
#  include "zensim/omp/execution/ExecutionPolicy.hpp"
#endif
#include "zensim/ZpcFunctional.hpp"

namespace zs {

#if ZS_ENABLE_OPENMP
#  define DEF_POLICY                            \
    constexpr auto space = execspace_e::openmp; \
    auto pol = omp_exec();
#else
#  define DEF_POLICY                          \
    constexpr auto space = execspace_e::host; \
    auto pol = seq_exec();
#endif

  struct CustomKey {
    u64 k;
    double v;
    constexpr bool operator<(const CustomKey &rhs) const noexcept { return k < rhs.k; }
  };

  inline Vector<int> gen_rnd_ints(size_t n, u32 upper = detail::deduce_numeric_max<int>()) {
    DEF_POLICY
#if 0
    std::random_device rd;
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(lower, upper);
    for (auto &v : vals) v = distrib(gen);
#endif
    Vector<int> vals(n);
    pol(enumerate(vals), [upper](size_t i, int &v) {
      u64 sd = i;
      v = zs::PCG::pcg32_random_r(sd, 1442695040888963407ull) % upper;
    });
    return vals;
  }
  Vector<float> gen_rnd_floats(size_t n) {
    DEF_POLICY
    Vector<float> vals(n);
    pol(enumerate(vals), [](size_t i, float &v) {
      u64 sd = i;
      v = 1.f * zs::PCG::pcg32_random_r(sd, 1442695040888963407ull) / limits<u32>::max();
    });
    return vals;
  }

  inline TileVector<float, 32> gen_rnd_tv_floats(size_t n) {
    DEF_POLICY
    TileVector<float, 32> vals({{"a", 3}, {"b", 2}, {"c", 1}}, n);
    pol(enumerate(range(vals, "b")), [](size_t i, float &v) {
      u64 sd = i;
      v = 1.f * zs::PCG::pcg32_random_r(sd, 1442695040888963407ull) / limits<u32>::max();
    });
    return vals;
  }
  inline TileVector<int, 32> gen_rnd_tv_ints(size_t n,
                                             u32 upper = detail::deduce_numeric_max<int>()) {
    DEF_POLICY
    TileVector<int, 32> vals({{"a", 3}, {"b", 2}, {"c", 1}}, n);
    pol(enumerate(range(vals, "b")), [upper](size_t i, int &v) {
      u64 sd = i;
      v = zs::PCG::pcg32_random_r(sd, 1442695040888963407ull) % upper;
    });
    return vals;
  }

}  // namespace zs