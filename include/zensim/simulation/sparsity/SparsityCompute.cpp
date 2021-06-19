#include "SparsityCompute.tpp"

#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

namespace zs {

  template GeneralHashTable partition_for_particles<execspace_e::host>(GeneralParticles&, float, int);

/*
  GeneralHashTable partition_for_particles(GeneralParticles &particles, float dx, int blocklen) {
    auto mh = match([](auto &p) { return p.handle(); })(particles);
    exec_tags execTag = suggest_exec_space(mh);
    HashTable<i32, 3, int> ret{match([](auto &particles) { return particles.size(); })(particles)
                                   / blocklen / blocklen / blocklen,
                               mh.memspace(), mh.devid()};
    match([&](auto execTag) -> GeneralHashTable{
      using ExecTag = remove_cvref_t<execTag>;
      auto execPol = par_exec(execTag);
      if constexpr (is_same_v<ExecTag, cuda_exec_tag> || is_same_v<ExecTag, hip_exec_tag>)
        execPol.device(mh.devid());

      execPol({ret._tableSize}, CleanSparsity{execTag, ret});
      match([&ret, &execTag, &execPol, dx, blocklen](auto &p) {
        execPol({p.size()}, ComputeSparsity{execTag, dx, blocklen, ret, p.X});
      })(particles);
    }, [](...) {})(execTag);
    return ret;

#if 0
    auto mh = match([](auto &p) { return p.handle(); })(particles);
    if (mh.onHost())
      return partition_for_particles_host_impl(particles, dx, blocklen);
    else if (mh.devid() >= 0)
      return partition_for_particles_cuda_impl(particles, dx, blocklen);
    else
      throw std::runtime_error(fmt::format("[partition_for_particles] scenario not implemented\n"));
#endif
  }
*/

}  // namespace zs