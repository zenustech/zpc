#include "../utils/initialization.hpp"
#include "../utils/parallel_primitives.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/types/Iterator.h"

int main() {
  using namespace zs;
  auto pol = cuda_exec();
  auto reduction = [&pol](size_t n) {
    // Vector<int> vals = gen_rnd_ints(n, make_monoid(getmin<int>()).identity());
    auto vals = gen_rnd_tv_ints(n, make_monoid(getmin<int>()).identity());
    vals = vals.clone({memsrc_e::device});
    if (!test_reduction(pol, range(vals, "b"), getmax<int>()))
      throw std::runtime_error("getmax<int> failed");
    if (!test_reduction(pol, range(vals, "b"), getmin<int>()))
      throw std::runtime_error("getmin<int> failed");
    vals = gen_rnd_tv_ints(n, 100);
    vals = vals.clone({memsrc_e::device});
    if (!test_reduction(pol, range(vals, "b"), plus<int>()))
      throw std::runtime_error("plus<int> failed");
  };
  for (int i = 0; i != 10; ++i) {
    reduction(1);
    reduction(2);
    reduction(7);
    reduction(16);
    reduction(128);
    reduction(1024);
    reduction(2000000);
  }
  return 0;
}