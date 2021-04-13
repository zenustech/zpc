#include "Simulator.hpp"

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

}  // namespace zs