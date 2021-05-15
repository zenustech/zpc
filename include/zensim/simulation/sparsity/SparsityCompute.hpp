#pragma once
#include "zensim/container/HashTable.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structurefree.hpp"

namespace zs {

  GeneralHashTable partition_for_particles_host_impl(GeneralParticles& particles, float dx,
                                                     int blocklen);
  GeneralHashTable partition_for_particles(GeneralParticles& particles, float dx, int blocklen);

}  // namespace zs