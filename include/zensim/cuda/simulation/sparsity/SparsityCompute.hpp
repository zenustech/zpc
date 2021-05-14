#pragma once
#include "zensim/container/HashTable.hpp"
#include "zensim/geometry/Structurefree.hpp"

namespace zs {

  GeneralHashTable partition_for_particles_cuda_impl(const GeneralParticles &particles, float dx,
                                                     int blocklen);

}  // namespace zs