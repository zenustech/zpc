#pragma once
#include <zensim/execution/ExecutionPolicy.hpp>

#include "zensim/container/Structurefree.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"

namespace zs {

  template <typename T, int d> bool convertParticles(Particles<T, d> &particles) {
    if (particles.space() == memsrc_e::cuda) {
      particles.particleBins = TileVector<f32, 32>{{"mass", "pos", "vel", "F", "C", "J", "logjp"},
                                                   particles.size(),
                                                   particles.memspace(),
                                                   particles.devid()};
      auto cuPol = cuda_exec().device(particles.devid());
      cuPol({particles.size()},
            [parray = proxy<execspace_e::cuda>(particles),
             ptiles = particles.particleBins.self()] __device__(auto i) mutable {
              ptiles("mass", i) = parray.mass(i);
              ptiles.tuple<3>("pos", i) = parray.pos(i);
              ptiles.tuple<3>("vel", i) = parray.vel(i);
            });
      return true;
    }
    return false;
  }

}  // namespace zs