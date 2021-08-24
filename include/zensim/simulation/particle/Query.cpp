#include "Query.hpp"

#include "Query.tpp"
#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  template GeneralIndexBuckets index_buckets_for_particles<execspace_e::host>(
      const GeneralParticles &particles, float);

  template GeneralIndexBuckets index_buckets_for_particles<SequentialExecutionPolicy>(
      const SequentialExecutionPolicy &, const GeneralParticles &particles, float, float);

  GeneralIndexBuckets build_neighbor_list_impl(host_exec_tag, const GeneralParticles &particles,
                                               float dx) {
    return match([](const auto &pars) -> GeneralIndexBuckets {
      constexpr int dim = remove_cvref_t<decltype(pars)>::dim;
      IndexBuckets<dim> indexBuckets{};
      return indexBuckets;
    })(particles);
  }

  GeneralIndexBuckets build_neighbor_list(const GeneralParticles &particles, float dx) {
    memsrc_e memLoc = match([](auto &pars) { return pars.space(); })(particles);
    if (memLoc != memsrc_e::host)
      return build_neighbor_list(exec_cuda, particles, dx);
    else
      return build_neighbor_list(exec_seq, particles, dx);
  }

}  // namespace zs