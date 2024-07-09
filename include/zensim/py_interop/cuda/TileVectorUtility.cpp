#include "zensim/container/TileVector.hpp"
#include "zensim/cuda/Cuda.h"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"

extern "C" {

#define INSTANTIATE_TILE_VECTOR_CAPIS(T, L)                                               \
  ZPC_EXPORT void append_properties__##cuda##_##tv##_##T##_##L(                           \
      zs::CudaExecutionPolicy *ppol, zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v,  \
      const std::vector<zs::PropertyTag> *tags) {                                         \
    v->append_channels(*ppol, *tags);                                                     \
  }                                                                                       \
  ZPC_EXPORT void append_properties__##cuda##_##tv##_##T##_##L##_##virtual(               \
      zs::CudaExecutionPolicy * ppol, zs::TileVector<T, L, zs::ZSPmrAllocator<true>> * v, \
      const std::vector<zs::PropertyTag> *tags) {                                         \
    v->append_channels(*ppol, *tags);                                                     \
  }

INSTANTIATE_TILE_VECTOR_CAPIS(int, 8)
INSTANTIATE_TILE_VECTOR_CAPIS(int, 32)
INSTANTIATE_TILE_VECTOR_CAPIS(int, 64)
INSTANTIATE_TILE_VECTOR_CAPIS(int, 512)
INSTANTIATE_TILE_VECTOR_CAPIS(float, 8)
INSTANTIATE_TILE_VECTOR_CAPIS(float, 32)
INSTANTIATE_TILE_VECTOR_CAPIS(float, 64)
INSTANTIATE_TILE_VECTOR_CAPIS(float, 512)
INSTANTIATE_TILE_VECTOR_CAPIS(double, 8)
INSTANTIATE_TILE_VECTOR_CAPIS(double, 32)
INSTANTIATE_TILE_VECTOR_CAPIS(double, 64)
INSTANTIATE_TILE_VECTOR_CAPIS(double, 512)
}