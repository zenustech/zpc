#include "SparsityCompute.hpp"

namespace zs {

  template <execspace_e space>
  GeneralHashTable partition_for_particles(GeneralParticles& particles, float dx, int blocklen) {
    auto mh = match([](auto &p) { return p.handle(); })(particles);
    HashTable<i32, 3, int> ret{match([](auto &particles) { return particles.size(); })(particles)
                                   / blocklen / blocklen / blocklen,
                               mh.memspace(), mh.devid()};

    // exec_tags execTag = suggest_exec_space(mh);
    auto execTag = wrapv<space>{};
    auto execPol = par_exec(execTag);
    if constexpr (space == execspace_e::cuda || space == execspace_e::hip)
      execPol.device(mh.devid());

    execPol(range(ret._tableSize), CleanSparsity{execTag, ret});
    match([&ret, &execTag, &execPol, dx, blocklen](auto &p) {
      execPol(range(p.size()), ComputeSparsity{execTag, dx, blocklen, ret, p.X});
    })(particles);
    return ret;
  }

}