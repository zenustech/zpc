#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/container/Bht.hpp"
#include "zensim/cuda/Cuda.h"

extern "C" {

#define INSTANTIATE_BHT_CAPIS(Tn, Dim, Index, B)                                                 \
  ZPC_EXPORT void resize_container##__##cuda##_##bht##_##Tn##_##Dim##_##Index##_##B(             \
      zs::CudaExecutionPolicy *ppol, zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<false>> *v,   \
      zs::size_t newCapacity) {                                                                  \
    v->resize(*ppol, newCapacity);                                                               \
  }                                                                                              \
  ZPC_EXPORT void resize_container##__##cuda##_##bht##_##Tn##_##Dim##_##Index##_##B##_##virtual( \
      zs::CudaExecutionPolicy * ppol, zs::bht<Tn, Dim, Index, B, zs::ZSPmrAllocator<true>> * v,  \
      zs::size_t newCapacity) {                                                                  \
    v->resize(*ppol, newCapacity);                                                               \
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