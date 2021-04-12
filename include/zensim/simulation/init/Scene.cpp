#include "Scene.hpp"

#include <zensim/memory/MemoryResource.h>
#include <zensim/resource/Resource.h>

#include <filesystem>
#include <stdexcept>

#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/core.h"

namespace zs {

  namespace fs = std::filesystem;

  SceneBuilder Scene::create() { return {}; }

  BuilderForSceneParticle BuilderForScene::particle() {
    return BuilderForSceneParticle{this->target()};
  }
  BuilderForSceneMesh BuilderForScene::mesh() { return BuilderForSceneMesh{this->target()}; }

  BuilderForSceneParticle &BuilderForSceneParticle::addParticles(std::string fn, float dx,
                                                                 float ppc) {
    fs::path p{fn};
    ParticleModel positions;
    if (p.extension() == ".vdb")
      positions = sample_from_vdb_file(fn, dx, ppc);
    else if (p.extension() == ".obj")
      positions = sample_from_obj_file(fn, dx, ppc);
    else
      fmt::print(fg(fmt::color::red), "does not support format {}\n", fn);
    fmt::print(fg(fmt::color::green), "done sampling {} particles [{}] with (dx: {}, ppc: {})\n",
               positions.size(), fn, dx, ppc);
    if (positions.size()) particlePositions.push_back(std::move(positions));
    return *this;
  }

  BuilderForSceneParticle &BuilderForSceneParticle::setConstitutiveModel(
      constitutive_model_e model) {
    switch (model) {
      case constitutive_model_e::EquationOfState:
        config = EquationOfStateConfig{};
        break;
      case constitutive_model_e::NeoHookean:
        config = NeoHookeanConfig{};
        break;
      case constitutive_model_e::FixedCorotated:
        config = FixedCorotatedConfig{};
        break;
      case constitutive_model_e::VonMisesFixedCorotated:
        config = VonMisesFixedCorotatedConfig{};
        break;
      case constitutive_model_e::DruckerPrager:
        config = DruckerPragerConfig{};
        break;
      case constitutive_model_e::NACC:
        config = NACCConfig{};
        break;
      default:
        fmt::print(fg(fmt::color::red), "constitutive model not known!");
        break;
    }
    return *this;
  }

  BuilderForSceneParticle &BuilderForSceneParticle::output(std::string fn) {
    for (auto &&[id, positions] : zip(range(particlePositions.size()), particlePositions))
      write_partio<float, 3>(fn + std::to_string(id) + ".bgeo", positions);
    return *this;
  }
  BuilderForSceneParticle &BuilderForSceneParticle::push(MemoryHandle dst) {
    auto &scene = this->target();
    auto &particles = scene.particles;
    for (auto &positions : particlePositions) {
      Particles<f32, 3> dstParticles{positions.size(), dst.memspace(), dst.devid()};
      auto &rm = get_resource_manager().get();
      // rm.copy();
    }
    return *this;
  }

}  // namespace zs