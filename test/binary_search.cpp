#include "utils/binary_search.hpp"

#include "utils/initialization.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"

int main() {
  using namespace zs;
  // auto pol = seq_exec();
  auto pol = seq_exec();
  auto reduction = [&pol](size_t n) {
    // Vector<int> vals = gen_rnd_ints(n, make_monoid(getmin<int>()).identity()
    auto vals = gen_rnd_floats(n);
    sort_floats(vals);
    if (!test_binary_search(pol, vals)) throw std::runtime_error("binary search failed");
  };
  for (int i = 0; i != 10; ++i) {
    reduction(1);
    reduction(2);
    reduction(3);
    reduction(4);
    reduction(7);
    reduction(16);
    reduction(128);
    reduction(129);
    reduction(1024);
    reduction(50000);
    reduction(200000);
  }
  return 0;
}