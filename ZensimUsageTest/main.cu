#include <zensim/cuda/Cuda.h>
#include <zensim/types/Iterator.h>

#include <zensim/execution/ExecutionPolicy.hpp>

int main() {
  using namespace zs;
  omp_exec()(range(10), [](int i) { fmt::print("{}\n", i); });
  return 0;
}
