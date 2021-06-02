#include "SparsityCompute.hpp"

#include "zensim/container/Vector.hpp"
#include "zensim/cuda/container/HashTable.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/simulation/sparsity/SparsityOp.hpp"

namespace zs {

  GeneralHashTable partition_for_particles_cuda_impl(GeneralParticles &particles, float dx,
                                                     int blocklen) {
    auto mh = match([](auto &p) { return p.handle(); })(particles);
    HashTable<i32, 3, int> ret{match([](auto &particles) { return particles.size(); })(particles)
                                   / blocklen / blocklen / blocklen,
                               mh.memspace(), mh.devid()};
    auto cudaPol = cuda_exec().device(0);
    cudaPol({ret._tableSize}, CleanSparsity{exec_cuda, ret});

    match([&ret, &cudaPol, dx, blocklen](auto &p) {
      cudaPol({p.size()}, ComputeSparsity{exec_cuda, dx, blocklen, ret, p.X});
    })(particles);
    return ret;
  }

}  // namespace zs