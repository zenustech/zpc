#include "Simulator.hpp"

#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Collider.h"
#include "zensim/physics/SoundSpeedCfl.hpp"
#include "zensim/zpc_tpls/magic_enum/magic_enum.hpp"

namespace zs {

  MPMSimulatorBuilder MPMSimulator::create() { return {}; }

  BuilderForMPMSimulator& BuilderForMPMSimulator::addScene(Scene&& scene) {
    // constitutive model
    const int particleModelOffset = this->target().particles.size();
    for (auto&& [config, geomTag, localId] : scene.models)
      if (geomTag == Scene::model_e::Particle)
        this->target().models.emplace_back(config, particleModelOffset + localId);
    // particles
    for (size_t i = 0; i < scene.particles.size(); ++i)
      /// range-based for loop might not be safe after moved
      this->target().particles.push_back(std::move(scene.particles[i]));
    this->target().boundaries = std::move(scene.boundaries);
#if 0
    for (auto&& boundary : this->target().boundaries) {
      match([](auto& b)
                -> enable_if_type<is_levelset_boundary<remove_cvref_t<decltype(b)>>::value> {
                  b = b.clone(MemoryHandle{memsrc_e::um, -5});
                },
            [](...) {})(boundary);
    }
#endif
    return *this;
  }
  BuilderForMPMSimulator& BuilderForMPMSimulator::setSimOptions(const SimOptions& ops) {
    this->target().simOptions = ops;
    return *this;
  }
  BuilderForMPMSimulator& BuilderForMPMSimulator::setPrimaryGridType(grid_e gt) {
    this->gridType = gt;
    return *this;
  }

  BuilderForMPMSimulator::operator MPMSimulator() noexcept {
    std::vector<MemoryProperty> memDsts(0);
    std::vector<std::vector<std::tuple<zs::size_t, zs::size_t>>> groups(0);
    auto searchHandle = [&memDsts](MemoryLocation mloc) -> int {
      for (auto&& [id, entry] : zs::zip(zs::range(memDsts.size()), memDsts))
        if (mloc.memspace() == entry.memspace() && mloc.devid() == entry.devid()) return id;
      return -1;
    };
    {
      const auto dx = this->target().simOptions.dx;
      const auto cfl = this->target().simOptions.cfl;
      float defaultDt = detail::deduce_numeric_max<float>();
      for (auto&& [model, id] : this->target().models) {
        match([dx, cfl, &defaultDt](auto&& m) {
          if constexpr (!is_same_v<RM_CVREF_T(m), EquationOfStateConfig>) {
            auto v = evaluate_timestep_linear_elasticity(m.E, m.nu, m.rho, dx, cfl);
            if (v < defaultDt) defaultDt = v;
          }
        })(model);
      }
      this->target().evaluatedDt = defaultDt;
    }
    auto searchModel = [&models = this->target().models](size_t objId) {
      int modelid{0};
      for (auto&& [model, id] : models) {
        if (id == objId) return modelid;
        modelid++;
      }
      return -1;
    };
    std::vector<size_t> numParticles(0);
    size_t id = 0;
    this->target().buckets.resize(this->target().particles.size());  // new
    for (auto&& particles : this->target().particles) {
      match([&](auto& ps) {
        auto did = searchHandle(ps.memoryLocation());
        if (did == -1) {
          memDsts.push_back(MemoryProperty{ps.memoryLocation()});
          numParticles.push_back(ps.size());
          groups.emplace_back(std::move(std::vector<std::tuple<zs::size_t, zs::size_t>>{
              std::make_tuple(searchModel(id), id)}));
        } else {
          numParticles[did] += ps.size();
          groups[did].push_back(std::make_tuple(searchModel(id), id));
        }
      })(particles);
      id++;
    }
    fmt::print("target processor\n");
    for (auto mh : memDsts)
      fmt::print("[{}, {}] ", magic_enum::enum_name(mh.memspace()), static_cast<int>(mh.devid()));
    fmt::print("\ntotal num particles per processor\n");
    for (auto np : numParticles) fmt::print("{} ", np);
    fmt::print("\n");
    for (auto&& [groupid, groups] : zs::zip(zs::range(groups.size()), groups)) {
      fmt::print("group {}: ", groupid);
      for (auto&& [modelid, objid] : groups) fmt::print("[{}, {}] ", modelid, objid);
      fmt::print("\n");
    }

    std::vector<size_t> numBlocks(numParticles.size());
    for (auto&& [dst, src] : zs::zip(numBlocks, numParticles)) dst = src / 8;
    for (auto&& [id, n] : zs::zip(range(numBlocks.size()), numBlocks))
      fmt::print("allocating {} blocks for partition {} in total!\n", n, id);

    /// particle model groups
    /// grid blocks, partitions
    this->target().grids.resize(memDsts.size());  // new
    this->target().partitions.resize(memDsts.size());
    this->target().maxVelSqrNorms.resize(memDsts.size());
    for (auto&& [memDst, nblocks, grids, partition, maxVel] :
         zs::zip(memDsts, numBlocks, this->target().grids, this->target().partitions,
                 this->target().maxVelSqrNorms)) {
      grids = Grids<f32, 3, 4>{{{"m", 1}, {"v", 3}, {"rhs", 3}},
                               target().simOptions.dx,
                               nblocks,
                               memDst.memspace(),
                               memDst.devid(),
                               gridType};
      partition = HashTable<i32, 3, int>{nblocks, memDst.memspace(), memDst.devid()};
      maxVel = Vector<float>{1, memDst.memspace(), memDst.devid()};
    }
    this->target().memDsts = std::move(memDsts);
    this->target().groups = std::move(groups);

    return std::move(this->target());
  }

}  // namespace zs