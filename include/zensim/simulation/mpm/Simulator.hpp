#pragma once
#include "zensim/simulation/init/Scene.hpp"
#include "zensim/types/BuilderBase.hpp"

namespace zs {

  struct MPMSimulatorBuilder;
  struct BuilderForMPMSimulator;
  struct MPMSimulator {
    /// advance
    ;
    /// construct
    static MPMSimulatorBuilder create();

  protected:
    friend struct BuilderForMPMSimulator;
    /// particle
    std::vector<GeneralParticles> particles;
    std::vector<std::tuple<ConstitutiveModelConfig, std::size_t>>
        models;  // (constitutive model, id)
    /// background grid
    GeneralGridBlocks gridBlocks;
    /// sparsity info
    // hash table
    /// transfer operator
    // apic/ flip
    /// simulation setup
    SimOptions simOptions;
  };

  struct BuilderForMPMSimulator : BuilderFor<MPMSimulator> {
    explicit BuilderForMPMSimulator(MPMSimulator& simulator)
        : BuilderFor<MPMSimulator>{simulator} {}

    BuilderForMPMSimulator& addScene(Scene&& scene);
    BuilderForMPMSimulator& setSimOptions(const SimOptions& ops);
  };

  struct MPMSimulatorBuilder : BuilderForMPMSimulator {
    MPMSimulatorBuilder() : BuilderForMPMSimulator{_simulator} {}

  protected:
    MPMSimulator _simulator;
  };

}  // namespace zs