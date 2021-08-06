#include "SparsityCompute.tpp"

#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  template GeneralHashTable partition_for_particles<execspace_e::host>(GeneralParticles&, float,
                                                                       int);

}  // namespace zs