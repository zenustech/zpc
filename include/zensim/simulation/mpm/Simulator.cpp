#include "Simulator.hpp"

#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  MPMSimulatorBuilder MPMSimulator::create() { return {}; }

  BuilderForMPMSimulator& BuilderForMPMSimulator::addScene(Scene&& scene) {
    // constitutive model
    const int particleModelOffset = this->target().particles.size();
    for (auto&& [config, geomTag, localId] : scene.models)
      if (geomTag == Scene::model_e::Particle)
        this->target().models.emplace_back(config, particleModelOffset + localId);
    // particles
    for (int i = 0; i < scene.particles.size(); ++i)
      /// range-based for loop might not be safe after moved
      this->target().particles.push_back(std::move(scene.particles[i]));
    return *this;
  }
  BuilderForMPMSimulator& BuilderForMPMSimulator::setSimOptions(const SimOptions& ops) {
    this->target().simOptions = ops;
    return *this;
  }

  BuilderForMPMSimulator::operator MPMSimulator() noexcept {
    std::map<ProcID, std::size_t> procMap{};
    std::vector<ProcID> memDsts(0);
    std::vector<std::size_t> numParticles(0);
    for (auto&& particles : this->target().particles) {
      match([&](auto& ps) {
        if (procMap.find(ps.devid()) == procMap.end()) {
          procMap.emplace(ps.devid(), memDsts.size());
          memDsts.push_back(ps.devid());
          numParticles.push_back(ps.size());
        } else
          numParticles[procMap[ps.devid()]] += ps.size();
      })(particles);
    }
    fmt::print("target processor\n");
    for (auto proc : memDsts) fmt::print("{} ", static_cast<int>(proc));
    fmt::print("\ntotal num particles per processor\n");
    for (auto np : numParticles) fmt::print("{} ", np);
    fmt::print("\n");

    std::vector<std::size_t> numBlocks(numParticles.size());
    for (auto&& [dst, src] : zs::zip(numBlocks, numParticles)) dst = src / 8 / 64;
    return std::move(this->target());
  }

}  // namespace zs