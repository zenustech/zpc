#pragma once
#include "zensim/simulation/init/Scene.hpp"
#include "zensim/types/BuilderBase.hpp"

namespace zs {

  template <typename PG> struct MPMSimulatorBuilder;
  template <typename PGrid> struct MPMSimulator {
    ;

  protected:
    friend struct MPMSimulatorBuilder<PGrid>;
    Scene scene;
    SimOptions simOptions;
  };

  template <typename PG> struct MPMSimulatorBuilder : BuilderFor<MPMSimulator<PG>> {
    MPMSimulatorBuilder() : BuilderFor<MPMSimulator<PG>>{simulator} {}

    MPMSimulatorBuilder& scene(SceneBuilder& sceneBuilder) { simulator.scene = sceneBuilder; }

    float dx;
    MPMSimulator<PG>& simulator;
  };

}  // namespace zs