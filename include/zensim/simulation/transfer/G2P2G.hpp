#pragma once
#include "Scheme.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/execution/Atomics.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structure.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/math/matrix/MatrixUtils.h"
#include "zensim/physics/ConstitutiveModel.hpp"

namespace zs {

  template <transfer_scheme_e, typename ConstitutiveModel, typename GridView, typename GridDofView,
            typename TableT, typename ParticlesT>
  struct G2P2GTransfer;

  template <execspace_e space, transfer_scheme_e scheme, typename Model, typename GridView,
            typename GridDofView, typename TableT, typename ParticlesT>
  G2P2GTransfer(wrapv<space>, wrapv<scheme>, float, Model, GridView, GridDofView, GridDofView,
                TableT, ParticlesT)
      -> G2P2GTransfer<scheme, Model, GridView, GridDofView, HashTableView<space, TableT>,
                       ParticlesView<space, ParticlesT>>;

  template <execspace_e space, transfer_scheme_e scheme, typename ModelT, typename GridView,
            typename GridDofView, typename TableT, typename ParticlesT>
  struct G2P2GTransfer<scheme, ModelT, GridView, GridDofView, HashTableView<space, TableT>,
                       ParticlesView<space, ParticlesT>> {
    using model_t = ModelT;  ///< constitutive model
    using grid_view_t = GridView;
    using grid_dof_view_t = GridDofView;
    using value_type = typename grid_dof_view_t::scalar_value_type;
    using partition_t = HashTableView<space, TableT>;
    using particles_t = ParticlesView<space, ParticlesT>;
    static_assert(particles_t::dim == partition_t::dim,
                  "[particle-partition-grid] dimension mismatch");

    explicit G2P2GTransfer(wrapv<space>, wrapv<scheme>, float dt, const ModelT& model,
                           grid_view_t grid, grid_dof_view_t x, grid_dof_view_t r, TableT& table,
                           ParticlesT& particles)
        : model{model},
          gridv{x},
          gridr{r},
          partition{proxy<space>(table)},
          particles{proxy<space>(particles)},
          dx{grid.dx},
          dt{dt} {}

    constexpr void operator()(typename particles_t::size_type parid) noexcept {
      value_type const dx_inv = (value_type)1 / dx;
      if constexpr (particles_t::dim == 3) {
        value_type const D_inv = 4.f * dx_inv * dx_inv;
        using ivec3 = vec<int, particles_t::dim>;
        using vec3 = vec<value_type, particles_t::dim>;
        using vec9 = vec<value_type, particles_t::dim * particles_t::dim>;
        using vec3x3 = vec<value_type, particles_t::dim, particles_t::dim>;
        vec3 pos{particles.pos(parid)};

        vec9 C{vec9::zeros()}, contrib{vec9::zeros()};
        auto arena = make_local_arena(dx, pos);
        for (auto loc : arena.range()) {
          auto [blockcoord, local_index]
              = unpack_coord_in_grid(arena.coord(loc), grid_view_t::side_length);

          auto blockno = partition.query(blockcoord);
          const auto cellid = grid_view_t::coord_to_cellid(local_index);

          auto xixp = arena.diff(loc);
          float W = arena.weight(loc);

          vec3 vi = gridv.get(blockno * grid_view_t::block_space() + cellid, vector_c);
          // vec3 vi = grid_block.pack<particles_t::dim>(1,
          // grid_view_t::coord_to_cellid(local_index));
          for (int d = 0; d < 9; ++d)
            C[d] += W * vi(d % particles_t::dim) * xixp(d / particles_t::dim) * D_inv;
        }

        // compute dPdF
        if constexpr (is_same_v<model_t, EquationOfStateConfig>) {
          float J = particles.J(parid);
          J = (1 + (C[0] + C[4] + C[8]) * dt) * J;
          // if (J < 0.1) J = 0.1;
          float vol = model.volume * J;
          float pressure = model.bulk;
          {
            float J2 = J * J;
            float J4 = J2 * J2;
            // pressure = pressure * (powf(J, -model.gamma) - 1.f);
            pressure = pressure * (1 / (J * J2 * J4) - 1);  // from Bow
          }
          contrib[0] = ((C[0] + C[0]) * model.viscosity - pressure) * vol;
          contrib[1] = (C[1] + C[3]) * model.viscosity * vol;
          contrib[2] = (C[2] + C[6]) * model.viscosity * vol;

          contrib[3] = (C[3] + C[1]) * model.viscosity * vol;
          contrib[4] = ((C[4] + C[4]) * model.viscosity - pressure) * vol;
          contrib[5] = (C[5] + C[7]) * model.viscosity * vol;

          contrib[6] = (C[6] + C[2]) * model.viscosity * vol;
          contrib[7] = (C[7] + C[5]) * model.viscosity * vol;
          contrib[8] = ((C[8] + C[8]) * model.viscosity - pressure) * vol;
        } else {
          vec9 oldF{particles.F(parid)}, tmp{}, F{};
          for (int d = 0; d < 9; ++d) tmp(d) = C[d] * dt + ((d & 0x3) ? 0.f : 1.f);
          matrixMatrixMultiplication3d(tmp.data(), oldF.data(), F.data());
          // particles.F(parid) = F;
          const auto [mu, lambda] = lame_parameters(model.E, model.nu);
          if constexpr (is_same_v<model_t, FixedCorotatedConfig>) {
            compute_stress_fixedcorotated(model.volume, mu, lambda, F, contrib);
          } else if constexpr (is_same_v<model_t, VonMisesFixedCorotatedConfig>) {
            compute_stress_vonmisesfixedcorotated(model.volume, mu, lambda, model.yieldStress, F,
                                                  contrib);
          } else {
            /// with plasticity additionally
            float logJp = particles.logJp(parid);
            if constexpr (is_same_v<model_t, DruckerPragerConfig>) {
              compute_stress_sand(model.volume, mu, lambda, model.cohesion, model.beta,
                                  model.yieldSurface, model.volumeCorrection, logJp, F, contrib);
            } else if constexpr (is_same_v<model_t, NACCConfig>) {
              compute_stress_nacc(model.volume, mu, lambda, model.bulk(), model.xi, model.beta,
                                  model.Msqr(), model.hardeningOn, logJp, F, contrib);
            }
          }
        }
        contrib *= D_inv;

        for (auto loc : arena.range()) {
          auto [blockcoord, local_index]
              = unpack_coord_in_grid(arena.coord(loc), grid_view_t::side_length);
          auto blockno = partition.query(blockcoord);
          // auto grid_block = gridr.block(blockno);
          auto xixp = arena.diff(loc);
          value_type W = arena.weight(loc);
          const auto cellid = grid_view_t::coord_to_cellid(local_index);
          const auto node = blockno * grid_view_t::block_space() + cellid;
          for (int d = 0; d != particles_t::dim; ++d)
            atomicAdd(
                &gridr.ref(node * particles_t::dim + d),
                W * (contrib[d] * xixp[0] + contrib[3 + d] * xixp[1] + contrib[6 + d] * xixp[2]));
        }
      }
    }

    model_t model;
    grid_dof_view_t gridv, gridr;
    partition_t partition;
    particles_t particles;
    value_type dx;
    float dt;
  };

}  // namespace zs