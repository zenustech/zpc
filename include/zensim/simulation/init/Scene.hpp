#include "Structure.hpp"
#include "Structurefree.hpp"
#include "zensim/geometry/LevelSet.h"
#include "zensim/math/Vec.h"
#include "zensim/physics/ConstitutiveModel.hpp"
#include "zensim/tpls/magic_enum.hpp"
#include "zensim/types/BuilderBase.hpp"

namespace zs {

  struct SceneBuilder;
  /// scene setup
  struct Scene {
    enum struct model_e : char { None = 0, LevelSet, Particle, Mesh, Nodes, Grid };
    using ModelHandle = std::pair<model_e, int>;
    // std::vector<GeneralLevelSet> levelsets;
    std::vector<GeneralParticles> particles;
    std::vector<GeneralMesh> meshes;
    std::vector<GeneralNodes> nodes;
    std::vector<GeneralGridBlocks> grids;
    std::array<ModelHandle, magic_enum::enum_integer(constitutive_model_e::NumConstitutiveModels)>
        elasticity_models;
    static SceneBuilder create();
  };

  struct BuilderForSceneParticle;
  struct BuilderForSceneMesh;
  struct BuilderForScene : BuilderFor<Scene> {
    explicit BuilderForScene(Scene &scene) : BuilderFor<Scene>{scene} {}
    BuilderForSceneParticle particle();
    BuilderForSceneMesh mesh();
  };

  struct SceneBuilder : BuilderForScene {
    SceneBuilder() : BuilderForScene{_scene} {}

  protected:
    Scene _scene;
  };

  struct BuilderForSceneParticle : BuilderForScene {
    explicit BuilderForSceneParticle(Scene &scene) : BuilderForScene{scene} {}
    /// particle positions
    BuilderForSceneParticle &addParticles(std::string fn, float dx, float ppc);
    // void addParticles(levelset, dx) {}

    /// constitutive models
    BuilderForSceneParticle &setConstitutiveModel(constitutive_model_e);

    /// push to scene
    BuilderForSceneParticle &push(MemoryHandle dst);
    /// check build status
    BuilderForSceneParticle &output(std::string fn);

    using ParticleModel = std::vector<std::array<float, 3>>;
  protected:
    std::vector<ParticleModel> particlePositions;
    ConstitutiveModelConfig config{EquationOfStateConfig{}};
  };
  struct BuilderForSceneMesh : BuilderForScene {
    explicit BuilderForSceneMesh(Scene &scene) : BuilderForScene{scene} {}
    BuilderForSceneMesh &addMesh(std::string fn) { return *this; }

    // std::vector<Mesh> meshes;
    ConstitutiveModelConfig config{EquationOfStateConfig{}};
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