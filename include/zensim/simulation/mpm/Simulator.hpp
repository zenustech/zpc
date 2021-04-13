#pragma once
#include "zensim/simulation/init/Scene.hpp"
#include "zensim/types/BuilderBase.hpp"

namespace zs {

  struct MPMSimulatorBuilder;
  struct MPMSimulator {
    ;

    static MPMSimulatorBuilder create();

  protected:
    SimOptions simOptions;
  };

  struct BuilderForMPMSimulatorOptions;
  struct BuilderForMPMSimulatorScene;
  struct BuilderForMPMSimulator : BuilderFor<MPMSimulator> {
    explicit BuilderForMPMSimulator(MPMSimulator& simulator)
        : BuilderFor<MPMSimulator>{simulator} {}

    BuilderForMPMSimulator& addScene(Scene& scene);
  };

  struct MPMSimulatorBuilder : BuilderForMPMSimulator {
    MPMSimulatorBuilder() : BuilderForMPMSimulator{_simulator} {}

  protected:
    float dx;
    MPMSimulator _simulator;
  };

}  // namespace zs