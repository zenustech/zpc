#include "SparsityCompute.hpp"

namespace zs {

  template <execspace_e space>
  GeneralHashTable partition_for_particles(GeneralParticles& particles, float dx, int blocklen) {
    return match([&](auto& p) -> GeneralHashTable {
      MemoryLocation mloc = p.memoryLocation();
      size_t keyCnt = p.size();
      constexpr int dim = remove_cvref_t<decltype(p)>::dim;
      for (auto d = dim; d--;) keyCnt /= blocklen;

      HashTable<i32, dim, int> ret{p.get_allocator(), keyCnt};

      // exec_tags execTag = suggest_exec_space(mh);
      auto execTag = wrapv<space>{};
      auto execPol = par_exec(execTag);
      if constexpr (is_device_execution_space<space>()) execPol.device(mloc.devid());

      execPol(range(ret._tableSize), CleanSparsity{execTag, ret});
      execPol(range(p.size()), ComputeSparsity{execTag, dx, blocklen, ret, p.attrVector("x")});
      return ret;
    })(particles);
  }

}  // namespace zs