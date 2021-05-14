#pragma once
#include "zensim/container/HashTable.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structurefree.hpp"

namespace zs {

  GeneralHashTable partition_for_particles_host_impl(const GeneralParticles& particles, float dx,
                                                     int blocklen);
  GeneralHashTable partition_for_particles(const GeneralParticles& particles, float dx,
                                           int blocklen);

}  // namespace zs