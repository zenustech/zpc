#pragma once
#include <zensim/execution/ExecutionPolicy.hpp>

#include "zensim/container/Structurefree.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"

namespace zs {

  template <typename T, int d> bool convertParticles(Particles<T, d> &particles) {
    if (particles.space() == memsrc_e::device) {
      constexpr auto dim = d;
      std::vector<zs::PropertyTag> properties{{"mass", 1}, {"pos", dim}, {"vel", dim}};
      if (particles.hasC()) properties.push_back({"C", dim * dim});
      if (particles.hasF()) properties.push_back({"F", dim * dim});
      if (particles.hasJ()) properties.push_back({"J", 1});
      if (particles.haslogJp()) properties.push_back({"logjp", 1});

      particles.particleBins
          = TileVector<f32, 32>{properties, particles.size(), particles.space(), particles.devid()};
      auto cuPol = cuda_exec().device(particles.devid());
      fmt::print("total channels {}\n", (int)particles.particleBins.numChannels());
#if 0
      cuPol({particles.size()},
            [parray = proxy<execspace_e::cuda>(particles),
             ptiles = proxy<execspace_e::cuda>(properties,
                                               particles.particleBins)] __device__(auto i) mutable {
              ptiles("mass", i) = parray.mass(i);
              ptiles.tuple<3>("pos", i) = parray.pos(i);
              ptiles.tuple<3>("vel", i) = parray.vel(i);
        if (ptiles.) ptiles.tuple<9>("F", i) = parray.vel(i);
            });
#endif
      return true;
    }
    return false;
  }

}  // namespace zs