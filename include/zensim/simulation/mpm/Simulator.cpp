#include "Simulator.hpp"

namespace zs {

  MPMSimulatorBuilder MPMSimulator::create() { return {}; }

  BuilderForMPMSimulator& BuilderForMPMSimulator::addScene(Scene& scene) { return *this; }

}  // namespace zs