#pragma once
#include "zensim/cuda/Cuda.h"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/types/SmallVector.hpp"

namespace zs {

#if 0
  template <typename T, int d> bool convertParticles(Particles<T, d> &particles) {
    if (particles.space() == memsrc_e::device || particles.space() == memsrc_e::um) {
      constexpr auto dim = d;
      std::vector<PropertyTag> properties{PropertyTag{"m", 1}, PropertyTag{"x", dim},
                                          PropertyTag{"v", dim}};
      if (particles.hasC()) properties.push_back(PropertyTag{"C", dim * dim});
      if (particles.hasF()) properties.push_back(PropertyTag{"F", dim * dim});
      if (particles.hasJ()) properties.push_back(PropertyTag{"J", 1});
      if (particles.haslogJp()) properties.push_back(PropertyTag{"logjp", 1});

      particles.particleBins
          = TileVector<f32, 32>{properties, particles.size(), particles.space(), particles.devid()};
      std::vector<SmallString> attribNames(properties.size());
      for (auto &&[dst, src] : zip(attribNames, properties)) dst = src.template get<0>();
      auto cuPol = cuda_exec().device(particles.devid());
      cuPol(
          {particles.size()},
          [dim, parray = proxy<execspace_e::cuda>(particles),
           ptiles = proxy<execspace_e::cuda>(
               attribNames,
               particles.particleBins)] __device__(typename Particles<T, d>::size_type i) mutable {
#  if 1
            if (i == 0) {
              printf("num total channels %d\n", (int)ptiles.numChannels());
              printf("mass channel %d offset: %d (%d)\n", (int)ptiles.propertyIndex("m"),
                     (int)ptiles._tagOffsets[ptiles.propertyIndex("m")], ptiles.hasProperty("m"));
              printf("pos channel %d offset: %d (%d)\n", (int)ptiles.propertyIndex("x"),
                     (int)ptiles._tagOffsets[ptiles.propertyIndex("x")], ptiles.hasProperty("x"));
              printf("vel channel %d offset: %d (%d)\n", (int)ptiles.propertyIndex("v"),
                     (int)ptiles._tagOffsets[ptiles.propertyIndex("v")], ptiles.hasProperty("v"));
              printf("F channel %d offset: %d (%d)\n", (int)ptiles.propertyIndex("F"),
                     (int)ptiles._tagOffsets[ptiles.propertyIndex("F")], ptiles.hasProperty("F"));
              printf("C channel %d offset: %d (%d)\n", (int)ptiles.propertyIndex("C"),
                     (int)ptiles._tagOffsets[ptiles.propertyIndex("C")], ptiles.hasProperty("C"));
              printf("J channel %d offset: %d (%d)\n", (int)ptiles.propertyIndex("J"),
                     (int)ptiles._tagOffsets[ptiles.propertyIndex("J")], ptiles.hasProperty("J"));
              printf("logJp channel %d offset: %d (%d)\n", (int)ptiles.propertyIndex("logjp"),
                     (int)ptiles._tagOffsets[ptiles.propertyIndex("logjp")], ptiles.hasProperty("logjp"));
            }
#  endif
            ptiles.template tuple<dim>("x",
                                       i);  // = vec<float, 3>{0.f, 1.f, 2.f};  // parray.pos(i);
            ptiles("m", i) = parray.mass(i);
            ptiles.template tuple<dim>("v", i) = parray.vel(i);
            if (ptiles.hasProperty("C")) ptiles.template tuple<dim * dim>("C", i) = parray.C(i);
            if (ptiles.hasProperty("F")) ptiles.template tuple<dim * dim>("F", i) = parray.F(i);
            if (ptiles.hasProperty("J")) ptiles("J", i) = parray.J(i);
            if (ptiles.hasProperty("logjp")) ptiles("logjp", i) = parray.logJp(i);
          });
      return true;
    }
    return false;
  }
#endif

}  // namespace zs