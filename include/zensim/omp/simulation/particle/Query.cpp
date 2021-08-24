#include "zensim/simulation/particle/Query.tpp"

#include "zensim/omp/execution/ExecutionPolicy.hpp"

namespace zs {

  template GeneralIndexBuckets index_buckets_for_particles<execspace_e::openmp>(
      const GeneralParticles &particles, float);

  template GeneralIndexBuckets index_buckets_for_particles<OmpExecutionPolicy>(
      const OmpExecutionPolicy &, const GeneralParticles &particles, float, float);

}  // namespace zs