#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/container/Bht.hpp"

extern "C" {

#define INSTANTIATE_BHT_CAPIS(Tn, Dim, Index, B)                                               \
  void resize_container##__##omp##_##bht##_##Tn##_##Dim##_##Index##_##B(                       \
      zs::OmpExecutionPolicy *ppol, zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<false>> *v,  \
      zs::size_t newCapacity) {                                                                \
    v->resize(*ppol, newCapacity);                                                             \
  }                                                                                            \
  void resize_container##__##omp##_##bht##_##Tn##_##Dim##_##Index##_##B##_##virtual(           \
      zs::OmpExecutionPolicy * ppol, zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<true>> * v, \
      zs::size_t newCapacity) {                                                                \
    v->resize(*ppol, newCapacity);                                                             \
  }

INSTANTIATE_BHT_CAPIS(int, 1, int, 32)
INSTANTIATE_BHT_CAPIS(int, 1, int, 16)
INSTANTIATE_BHT_CAPIS(int, 2, int, 32)
INSTANTIATE_BHT_CAPIS(int, 2, int, 16)
INSTANTIATE_BHT_CAPIS(int, 3, int, 32)
INSTANTIATE_BHT_CAPIS(int, 3, int, 16)
INSTANTIATE_BHT_CAPIS(int, 4, int, 32)
INSTANTIATE_BHT_CAPIS(int, 4, int, 16)
}