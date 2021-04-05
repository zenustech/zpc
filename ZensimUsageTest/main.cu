#include "zensim/cuda/Cuda.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/types/Iterator.h"

int main() {
  using namespace zs;
  omp_exec()(range(10), [](int i) { fmt::print("{}\n", i); });
  return 0;
}
