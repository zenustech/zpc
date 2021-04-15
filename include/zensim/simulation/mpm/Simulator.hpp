#pragma once
#include "zensim/container/HashTable.hpp"
#include "zensim/simulation/init/Scene.hpp"
#include "zensim/types/BuilderBase.hpp"

namespace zs {

  struct MPMSimulatorBuilder;
  struct BuilderForMPMSimulator;
  struct MPMSimulator {
    /// construct
    static MPMSimulatorBuilder create();

  protected:
    friend struct BuilderForMPMSimulator;
    friend struct MPMSimulatorBuilder;
    /// particle
    std::vector<GeneralParticles> particles;
    std::vector<std::tuple<ConstitutiveModelConfig, std::size_t>>
        models;  // (constitutive model, id)
    ///
    std::vector<ProcID> memDsts;
    /// background grid
    std::vector<GeneralGridBlocks> gridBlocks;
    /// sparsity info (hash table)
    std::vector<GeneralHashTable> hashTables;
    /// transfer operator
    // apic/ flip
    /// simulation setup
    SimOptions simOptions;
  };

  template <execspace_e, typename Simulator, typename = void> struct MPMSimulatorProxy;

  template <execspace_e ExecSpace> decltype(auto) proxy(MPMSimulator& simulator) {
    return MPMSimulatorProxy<ExecSpace, MPMSimulator>{simulator};
  }

  struct BuilderForMPMSimulator : BuilderFor<MPMSimulator> {
    explicit BuilderForMPMSimulator(MPMSimulator& simulator)
        : BuilderFor<MPMSimulator>{simulator} {}

    BuilderForMPMSimulator& addScene(Scene&& scene);
    BuilderForMPMSimulator& setSimOptions(const SimOptions& ops);

    operator MPMSimulator() noexcept;
  };

  struct MPMSimulatorBuilder : BuilderForMPMSimulator {
    MPMSimulatorBuilder() : BuilderForMPMSimulator{_simulator} {}

  protected:
    MPMSimulator _simulator;
  };

}  // namespace zs