#pragma once
#include "zensim/container/Structurefree.hpp"

namespace zs {

  template <typename ParticlesT> struct SetParticleAttribute;

  template <execspace_e space, typename ParticlesT> SetParticleAttribute(wrapv<space>, ParticlesT)
      -> SetParticleAttribute<ParticlesProxy<space, ParticlesT>>;

  template <execspace_e space, typename ParticlesT>
  struct SetParticleAttribute<ParticlesProxy<space, ParticlesT>> {
    using particles_t = ParticlesProxy<space, ParticlesT>;

    explicit SetParticleAttribute(wrapv<space>, ParticlesT& particles)
        : particles{proxy<space>(particles)} {}

    constexpr void operator()(typename particles_t::size_type parid) noexcept {
      if constexpr (particles_t::dim == 3) {
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) {
            if (particles.C(parid)[i][j] != 0)
              printf("parid %d, C(%d, %d): %e\n", (int)parid, i, j, particles.C(parid)[i][j]);
          }
      }
    }

    particles_t particles;
  };

}  // namespace zs