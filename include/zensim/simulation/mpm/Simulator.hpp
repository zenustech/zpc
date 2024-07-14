#pragma once
#include "zensim/container/HashTable.hpp"
#include "zensim/container/IndexBuckets.hpp"
#include "zensim/memory/MemoryResource.h"
#include "zensim/resource/Resource.h"
#include "zensim/simulation/init/Scene.hpp"
#include "zensim/types/BuilderBase.hpp"

namespace zs {

  struct MPMSimulatorBuilder;
  struct BuilderForMPMSimulator;
  struct MPMSimulator {
    /// construct
    static MPMSimulatorBuilder create();

    size_t numModels() const noexcept { return particles.size(); }
    size_t numPartitions() const noexcept { return partitions.size(); }
    float getMaxVel(int partI) const {
      float ret[1];
      Resource::copy(MemoryEntity{MemoryLocation{memsrc_e::host, -1}, ret},
           MemoryEntity{maxVelSqrNorms[partI].memoryLocation(),
                        const_cast<float*>(maxVelSqrNorms[partI].data())},
           sizeof(float));
      return ret[0];
    }
    float* maxVelPtr(int partI) { return maxVelSqrNorms[partI].data(); }
    const float* maxVelPtr(int partI) const { return maxVelSqrNorms[partI].data(); }

    /// particle
    std::vector<GeneralParticles> particles;
    std::vector<std::tuple<ConstitutiveModelConfig, zs::size_t>>
        models;  // (constitutive model, id)
    /// parallel execution helper
    std::vector<MemoryHandle> memDsts;
    std::vector<std::vector<std::tuple<zs::size_t, zs::size_t>>> groups;  // (model id, object id)
    /// background grid
    std::vector<GeneralGrids> grids;
    /// sparsity info (hash table)
    std::vector<GeneralHashTable> partitions;
    std::vector<Vector<float>> maxVelSqrNorms;
    /// transfer operator
    /// boundary
    std::vector<GeneralBoundary> boundaries;
    ///
    std::vector<GeneralIndexBuckets> buckets;
    // apic/ flip
    /// simulation setup
    SimOptions simOptions;
    float evaluatedDt;
  };

  template <execspace_e, typename Simulator, typename = void> struct MPMSimulatorView;

  template <execspace_e ExecSpace> decltype(auto) proxy(MPMSimulator& simulator) {
    return MPMSimulatorView<ExecSpace, MPMSimulator>{simulator};
  }

  struct BuilderForMPMSimulator : BuilderFor<MPMSimulator> {
    explicit BuilderForMPMSimulator(MPMSimulator& simulator)
        : BuilderFor<MPMSimulator>{simulator} {}

    BuilderForMPMSimulator& addScene(Scene&& scene);
    BuilderForMPMSimulator& setSimOptions(const SimOptions& ops);
    BuilderForMPMSimulator& setPrimaryGridType(grid_e gt);

    operator MPMSimulator() noexcept;
    grid_e gridType{grid_e::collocated};
  };

  struct MPMSimulatorBuilder : BuilderForMPMSimulator {
    MPMSimulatorBuilder() : BuilderForMPMSimulator{_simulator} {}

  protected:
    MPMSimulator _simulator;
  };

}  // namespace zs