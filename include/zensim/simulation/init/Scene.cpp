#include "Scene.hpp"

#include <zensim/geometry/AnalyticLevelSet.h>
#include <zensim/memory/MemoryResource.h>
#include <zensim/resource/Resource.h>

#include <filesystem>
#include <stdexcept>

#include "zensim/geometry/GeometrySampler.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/math/Vec.h"
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
    ParticleModel positions{};
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
  BuilderForSceneParticle &BuilderForSceneParticle::addCuboid(std::vector<float> mi,
                                                              std::vector<float> ma, float dx,
                                                              float ppc) {
    using ALS = AnalyticLevelSet<analytic_geometry_e::Cuboid, float, 3>;
    ParticleModel positions{};
    if (mi.size() == 3 && ma.size() == 3)
      positions = sampleFromLevelSet(
          ALS{vec<float, 3>{mi[0], mi[1], mi[2]}, vec<float, 3>{ma[0], ma[1], ma[2]}}, dx, ppc);
    else
      fmt::print(fg(fmt::color::red), "cuboid build config dimension error ({}, {})\n", mi.size(),
                 ma.size());
    fmt::print(
        fg(fmt::color::green),
        "done sampling {} particles [cuboid ({}, {}, {}) - ({}, {}, {})] with (dx: {}, ppc: {})\n",
        positions.size(), mi[0], mi[1], mi[2], ma[0], ma[1], ma[2], dx, ppc);
    if (positions.size()) particlePositions.push_back(std::move(positions));
    return *this;
  }
  BuilderForSceneParticle &BuilderForSceneParticle::addCube(std::vector<float> c, float len,
                                                            float dx, float ppc) {
    std::vector<float> mi{c[0] - len / 2, c[1] - len / 2, c[2] - len / 2};
    std::vector<float> ma{c[0] + len / 2, c[1] + len / 2, c[2] + len / 2};
    return addCuboid(mi, ma, dx, ppc);
  }
  BuilderForSceneParticle &BuilderForSceneParticle::addSphere(std::vector<float> c, float r,
                                                              float dx, float ppc) {
    using ALS = AnalyticLevelSet<analytic_geometry_e::Sphere, float, 3>;
    ParticleModel positions{};
    if (c.size() == 3)
      positions = sampleFromLevelSet(ALS{vec<float, 3>{c[0], c[1], c[2]}, r}, dx, ppc);
    else
      fmt::print(fg(fmt::color::red), "sphere build config dimension error center{}\n", c.size());
    fmt::print(fg(fmt::color::green),
               "done sampling {} particles [sphere ({}, {}, {}), {}] with (dx: {}, ppc: {})\n",
               positions.size(), c[0], c[1], c[2], r, dx, ppc);
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
    displayConfig(config);
    for (auto &&[id, positions] : zip(range(particlePositions.size()), particlePositions))
      write_partio<float, 3>(fn + std::to_string(id) + ".bgeo", positions);
    return *this;
  }
  BuilderForSceneParticle &BuilderForSceneParticle::commit(MemoryHandle dst) {
    auto &scene = this->target();
    auto &dstParticles = scene.particles;
    for (auto &positions : particlePositions) {
      Particles<f32, 3> pos{positions.size(), dst.memspace(), dst.devid()};
      auto &rm = get_resource_manager().get();
      // rm.copy();
    }
    return *this;
  }

}  // namespace zs