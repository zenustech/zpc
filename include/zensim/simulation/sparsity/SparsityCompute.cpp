#include "SparsityCompute.hpp"

#include "zensim/execution/ExecutionPolicy.hpp"
#if ZS_ENABLE_CUDA
#  include "zensim/cuda/simulation/sparsity/SparsityCompute.hpp"
#endif

namespace zs {

  GeneralHashTable partition_for_particles_host_impl(const GeneralParticles &particles, float dx,
                                                     int blocklen) {
    throw std::runtime_error(
        fmt::format("copy operation backend {} for [{}, {}, {}] -> [{}, {}, {}] not implemented\n",
                    get_execution_space_tag(execspace_e::host)));
  }

  GeneralHashTable partition_for_particles(const GeneralParticles &particles, float dx,
                                           int blocklen) {
    auto mh = match([](auto &p) { return p.handle(); })(particles);
    if (mh.onHost())
      return partition_for_particles_host_impl(particles, dx, blocklen);
    else if (mh.devid() >= 0)
      return partition_for_particles_cuda_impl(particles, dx, blocklen);
    else
      throw std::runtime_error(fmt::format("[partition_for_particles] scenario not implemented\n"));
  }

}  // namespace zs