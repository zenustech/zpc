#include "Structure.hpp"
#include "Structurefree.hpp"
#include "zensim/geometry/LevelSet.h"
#include "zensim/math/Vec.h"
#include "zensim/physics/ConstitutiveModel.hpp"
#include "zensim/tpls/magic_enum.hpp"
#include "zensim/types/BuilderBase.hpp"

namespace zs {

  /// scene setup
  struct Scene {
    enum class model_e : char { None = 0, LevelSet, Particle, Mesh, Nodes, Grid };
    using ModelHandle = std::pair<model_e, int>;
    // std::vector<GeneralLevelSet> levelsets;
    std::vector<GeneralParticles> particles;
    std::vector<GeneralMesh> meshes;
    std::vector<GeneralNodes> nodes;
    std::vector<GeneralGridBlocks> grids;
    std::array<ModelHandle, *magic_enum::enum_index(NumConstitutiveModels)> elasticity_models;
  };

  struct BuilderForSceneParticle;
  struct BuilderForSceneMesh;
  struct BuilderForScene : BuilderFor<Scene> {
    explicit BuilderForScene(Scene &scene) : BuilderFor<Scene>{scene} {}
    BuilderForSceneParticle particle() const;
    BuilderForSceneMesh mesh() const;
  };

  struct SceneBuilder : BuilderForScene {
    SceneBuilder() : BuilderForScene{_scene} {}

  protected:
    Scene _scene;
  };

  struct BuilderForSceneParticle : BuilderForScene {
    explicit BuilderForSceneParticle(Scene &scene) : BuilderForScene{scene} {}
    BuilderForSceneParticle &addParticles(std::string fn) { return *this; }
    /// void addParticles(levelset, dx) {}
  };
  struct BuilderForSceneMesh : BuilderForScene {
    explicit BuilderForSceneMesh(Scene &scene) : BuilderForScene{scene} {}
    BuilderForSceneMesh &addMesh(std::string fn) { return *this; }
  };

  /// simulator setup
  struct SimOptions {
    float fps;
    float dt;
  };

  struct BuilderForSimOptions : BuilderFor<SimOptions> {
    BuilderForSimOptions(SimOptions &simOptions) : BuilderFor<SimOptions>{simOptions} {}
    BuilderForSimOptions &fps(float v) {
      this->object.fps = v;
      return *this;
    }
    BuilderForSimOptions &dt(float v) {
      this->object.dt = v;
      return *this;
    }
  };
  struct SimOptionsBuilder : BuilderForSimOptions {
    SimOptionsBuilder() : BuilderForSimOptions{_simOptions} {}

  protected:
    SimOptions _simOptions;
  };

}  // namespace zs