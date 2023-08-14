#pragma once
#include "zensim/container/HashTable.hpp"
#include "zensim/geometry/Structure.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/math/linear/LinearOperators.hpp"
#include "zensim/simulation/mpm/Simulator.hpp"
#include "zensim/simulation/transfer/G2P2G.hpp"

namespace zs {

  struct ImplicitMPMSystem {
    ImplicitMPMSystem(MPMSimulator& simulator, float dt, size_t partI)
        : simulator{simulator}, partI{partI}, dt{dt} {}

    template <typename DofA, typename DofB, typename DofC, typename DofD>
    struct ForceDtSqrPlusMass {
      using Index = typename DofA::size_type;
      static constexpr int dim = DofA::dim;
      ForceDtSqrPlusMass(DofA a, DofB b, DofC c, DofD d, float dt)
          : f{a}, m{b}, Ax{c}, dv{d}, dt{dt} {}

      constexpr void operator()(Index i) {
        if (auto mass = m.get(i / dim); mass > 0)
          Ax.set(i, (f.get(i, scalar_c) * dt * dt + mass) * dv.get(i, scalar_c));
      }
      DofA f;
      DofB m;
      DofC Ax;
      DofD dv;
      float dt;
    };
    template <class ExecutionPolicy, typename In, typename Out>
    void multiply(ExecutionPolicy&& policy, In&& in, Out&& out) {
      constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
      constexpr auto execTag = wrapv<space>{};
      auto mh = simulator.memDsts[partI];
      for (auto&& [modelId, objId] : simulator.groups[partI]) {
        auto& [model, objId_] = simulator.models[modelId];
        assert_with_msg(objId_ == objId, "[MPMSimulator] model-object id conflicts, error build");
        if (objId_ != objId) throw std::runtime_error("WTF???");
        match(
            [&, this, did = mh.devid()](
                auto& constitutiveModel, auto& partition, auto& obj,
                auto& grids) -> enable_if_type<RM_CVREF_T(obj)::dim == RM_CVREF_T(partition)::dim
                                                 && RM_CVREF_T(obj)::dim == RM_CVREF_T(grids)::dim
                                                 && RM_CVREF_T(obj)::dim == RM_CVREF_T(in)::dim> {
              policy(range(out.numEntries()), DofFill{out, 0});
              // compute f_i (out)
              policy(range(obj.size()),
                     G2P2GTransfer{execTag, wrapv<transfer_scheme_e::apic>{}, dt, constitutiveModel,
                                   proxy<space>(grids.grid()), in, out, partition, obj});
              // update v_i
              auto gridm = dof_view<space, 1>(grids.grid(), "m", 0);
              policy(range(in.numEntries()), ForceDtSqrPlusMass{out, gridm, out, in, dt});
            },
            [](...) {})(model, simulator.partitions[partI], simulator.particles[objId],
                        simulator.grids[partI]);
      }
    }

    template <typename ColliderView, typename TableView, typename GridView, typename GridDofView>
    struct Projector {
      using grid_t = GridView;
      using dof_index_t = typename grid_t::size_type;
      using value_type = typename grid_t::value_type;

      Projector(ColliderView col, TableView part, GridView grid, GridDofView dof)
          : collider{col}, partition{part}, grid{grid}, dof{dof} {}

      constexpr void operator()(dof_index_t nodei) {
        auto blockid = nodei / grid_t::block_space();
        auto cellid = nodei % grid_t::block_space();
        auto blockkey = partition._activeKeys[blockid];

        if (grid[blockid](0, cellid) > 0) {
          // auto vel = block.pack<GridDofView::dim>(1, cellid);
          auto vel = dof.get(nodei, vector_c);
          auto pos = (blockkey * (value_type)grid_t::side_length + grid_t::cellid_to_coord(cellid))
                     * grid.dx;

          collider.resolveCollision(pos, vel);

          dof.set(nodei, vel);
          // block.set(1, cellid, vel);
        } else {  // clear non-dof nodes as well
          using V = decltype(dof.get(0, vector_c));
          dof.set(nodei, V::zeros());
        }
      }

      ColliderView collider;
      TableView partition;
      GridView grid;
      GridDofView dof;
    };

    template <class ExecutionPolicy, typename InOut>
    void project(ExecutionPolicy&& policy, InOut&& inout) {
      constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
      auto mh = simulator.memDsts[partI];
      assert_with_msg(mh.devid() >= 0, "[MPMSimulator] should not put data on host");
      for (auto& boundary : simulator.boundaries) {
        match(
            [&, did = mh.devid()](auto& collider, auto& partition, auto& grids)
                -> enable_if_type<RM_CVREF_T(collider)::dim == RM_CVREF_T(partition)::dim
                                    && RM_CVREF_T(collider)::dim == RM_CVREF_T(grids)::dim
                                    && RM_CVREF_T(collider)::dim == RM_CVREF_T(inout)::dim> {
              // fmt::print("[gpu {}]\tprojecting {} grid blocks, dof dim: {}\n", (int)did,
              //            partition.size(), RM_CVREF_T(inout)::dim);
              if constexpr (is_levelset_boundary<RM_CVREF_T(collider)>::value)
                policy(range((size_t)inout.numEntries()
                             / remove_cvref_t<decltype(collider)>::dim),
                       Projector{Collider{proxy<space>(collider.levelset), collider.type},
                                 proxy<space>(partition), proxy<space>(grids.grid()), inout});
              else {
                policy(range((size_t)inout.numEntries()
                             / remove_cvref_t<decltype(collider)>::dim),
                       Projector{collider, proxy<space>(partition), proxy<space>(grids.grid()),
                                 inout});
              }
            },
            [](...) {})(boundary, simulator.partitions[partI], simulator.grids[partI]);
      }
    }

    template <typename DofA, typename DofB, typename DofC> struct DivPernodeMass {
      using Index = typename DofA::size_type;
      static constexpr auto dim = DofC::dim;
      DivPernodeMass(DofA a, DofB b, DofC c) : a{a}, b{b}, c{c} {}

      constexpr void operator()(Index i) {
        if (auto mass = b.get(i / dim); mass > 0) c.set(i, a.get(i, scalar_c) / mass);
      }
      DofA a;
      DofB b;
      DofC c;
    };

    template <class ExecutionPolicy, typename In, typename Out>
    void precondition(ExecutionPolicy&& policy, In&& in, Out&& out) {
      constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
      auto mh = simulator.memDsts[partI];
      assert_with_msg(mh.devid() >= 0, "[MPMSimulator] should not put data on host");
      match(
          [&, did = mh.devid()](auto& partition, auto& grids)
              -> enable_if_type<RM_CVREF_T(partition)::dim == RM_CVREF_T(grids)::dim> {
            fmt::print("[gpu {}]\tpreconditioning {} grid blocks\n", (int)did, partition.size());
            auto gridm = dof_view<space, 1>(grids.grid(), "m", 0);
            policy(range(out.numEntries()), DivPernodeMass{in, gridm, out});
          },
          [](...) {})(simulator.partitions[partI], simulator.grids[partI]);
    }

    MPMSimulator& simulator;
    size_t partI;
    float dt;
  };

}  // namespace zs