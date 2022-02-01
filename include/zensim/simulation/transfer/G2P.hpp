#pragma once
#include "Scheme.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structure.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/math/matrix/MatrixUtils.h"
#include "zensim/physics/ConstitutiveModel.hpp"

namespace zs {

  template <transfer_scheme_e, typename ConstitutiveModel, typename GridBlocksT, typename TableT,
            typename ParticlesT>
  struct G2PTransfer;

  template <execspace_e space, transfer_scheme_e scheme, typename Model, typename T, int d, auto l,
            typename TableT, typename ParticlesT>
  G2PTransfer(wrapv<space>, wrapv<scheme>, float, Model, Grids<T, d, l>, TableT, ParticlesT)
      -> G2PTransfer<scheme, Model, GridsView<space, Grids<T, d, l>>, HashTableView<space, TableT>,
                     ParticlesView<space, ParticlesT>>;

  template <execspace_e space, transfer_scheme_e scheme, typename ModelT, typename GridsT,
            typename TableT, typename ParticlesT>
  struct G2PTransfer<scheme, ModelT, GridsView<space, GridsT>, HashTableView<space, TableT>,
                     ParticlesView<space, ParticlesT>> {
    using model_t = ModelT;  ///< constitutive model
    using grids_t = GridsView<space, GridsT>;
    using value_type = typename grids_t::value_type;
    using partition_t = HashTableView<space, TableT>;
    using particles_t = ParticlesView<space, ParticlesT>;
    static_assert(particles_t::dim == partition_t::dim && particles_t::dim == grids_t::dim,
                  "[particle-partition-grid] dimension mismatch");

    explicit G2PTransfer(wrapv<space>, wrapv<scheme>, float dt, const ModelT& model, GridsT& grids,
                         TableT& table, ParticlesT& particles)
        : model{model},
          grids{proxy<space>(grids)},
          partition{proxy<space>(table)},
          particles{proxy<space>(particles)},
          dt{dt} {}

    constexpr void operator()(typename particles_t::size_type parid) noexcept {
      value_type const dx = grids._dx;
      value_type const dx_inv = (value_type)1 / dx;
      if constexpr (particles_t::dim == 3) {
        value_type const D_inv = 4.f * dx_inv * dx_inv;
        using ivec3 = vec<int, particles_t::dim>;
        using vec3 = vec<value_type, particles_t::dim>;
        using vec9 = vec<value_type, particles_t::dim * particles_t::dim>;
        using vec3x3 = vec<value_type, particles_t::dim, particles_t::dim>;
        vec3 pos{particles.pos(parid)};
        vec3 vel{vec3::zeros()};

        vec9 C{vec9::zeros()};
        auto arena = make_local_arena(dx, pos);
        for (auto loc : arena.range()) {
          auto [grid_block, local_index]
              = unpack_coord_in_grid(arena.coord(loc), grids_t::side_length, partition, grids);
          auto xixp = arena.diff(loc);
          float W = arena.weight(loc);

          vec3 vi = grid_block.pack<particles_t::dim>(1, grids_t::coord_to_cellid(local_index));
          vel += vi * W;
          for (int d = 0; d < 9; ++d) C[d] += W * vi(d % 3) * xixp(d / 3) * D_inv;
        }
        pos += vel * dt;

        if constexpr (is_same_v<model_t, EquationOfStateConfig>) {
          float J = particles.J(parid);
          J = (1 + (C[0] + C[4] + C[8]) * dt) * J;
          // if (J < 0.1) J = 0.1;
          particles.J(parid) = J;
        } else {
          vec9 oldF{particles.F(parid)}, tmp{}, F{};
          for (int d = 0; d < 9; ++d) tmp(d) = C[d] * dt + ((d & 0x3) ? 0.f : 1.f);
          matrixMatrixMultiplication3d(tmp.data(), oldF.data(), F.data());
          particles.F(parid) = F;
        }
        particles.pos(parid) = pos;
        particles.vel(parid) = vel;
        particles.C(parid) = C;
      }
    }

    model_t model;
    grids_t grids;
    partition_t partition;
    particles_t particles;
    float dt;
  };

}  // namespace zs