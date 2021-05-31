#include "Query.hpp"

// #include "zensim/cuda/execution/CudaExecutionPolicy.cuh"

namespace zs {

  GeneralIndexBuckets build_neighbor_list_impl(cuda_exec_tag, const GeneralParticles &particles,
                                               float dx) {
    return {};
  }

}  // namespace zs