#pragma once
#include "zensim/container/Structurefree.hpp"
#include "zensim/cuda/DeviceUtils.cuh"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/physics/ConstitutiveModel.hpp"
#include "zensim/simulation/transfer/P2G.hpp"

namespace zs {

  template <transfer_scheme_e scheme, typename ModelT, typename ParticlesT, typename TableT,
            typename GridBlocksT>
  struct P2GTransfer<scheme, ModelT, ParticlesProxy<execspace_e::cuda, ParticlesT>,
                     HashTableProxy<execspace_e::cuda, TableT>,
                     GridBlocksProxy<execspace_e::cuda, GridBlocksT>> {
    using model_t = ModelT;  ///< constitutive model
    using particles_t = ParticlesProxy<execspace_e::cuda, ParticlesT>;
    using partition_t = HashTableProxy<execspace_e::cuda, TableT>;
    using gridblocks_t = GridBlocksProxy<execspace_e::cuda, GridBlocksT>;
    using gridblock_t = typename gridblocks_t::block_t;
    static_assert(particles_t::dim == partition_t::dim && particles_t::dim == gridblocks_t::dim,
                  "[particle-partition-grid] dimension mismatch");

    explicit P2GTransfer(wrapv<execspace_e::cuda>, wrapv<scheme>, float dt, const ModelT& model,
                         ParticlesT& particles, TableT& table, GridBlocksT& gridblocks)
        : model{model},
          particles{proxy<execspace_e::cuda>(particles)},
          partition{proxy<execspace_e::cuda>(table)},
          gridblocks{proxy<execspace_e::cuda>(gridblocks)},
          dt{dt} {}

    constexpr float dxinv() const {
      return static_cast<decltype(gridblocks._dx.asFloat())>(1.0) / gridblocks._dx.asFloat();
    }

    __forceinline__ __device__ void operator()(typename particles_t::size_type parid) noexcept {
      if constexpr (particles_t::dim == 3
                    && std::is_same_v<
                        model_t,
                        EquationOfStateConfig> && sizeof(typename gridblock_t::value_type) == 4) {
        float const dx = gridblocks._dx.asFloat();
        float const dx_inv = dxinv();
        float const D_inv = 4.f * dx_inv * dx_inv;
        using ivec3 = vec<int, 3>;
        using vec3 = vec<float, 3>;
        using vec9 = vec<float, 9>;
        using vec3x3 = vec<float, 3, 3>;
        vec3 local_pos{particles.pos(parid)[0], particles.pos(parid)[1], particles.pos(parid)[2]};
        vec3 vel{particles.vel(parid)[0], particles.vel(parid)[1], particles.vel(parid)[2]};
        float J = particles.J(parid);
        float mass = particles.mass(parid);
        float vol = model.volume * J;
        float pressure = model.bulk * (powf(J, -model.gamma) - 1.f);
        // float pressure = model.bulk * (1 / J / J / J / J / J / J / J - 1);

        vec9 contrib,
            C{particles.C(parid)[0][0], particles.C(parid)[1][0], particles.C(parid)[2][0],
              particles.C(parid)[0][1], particles.C(parid)[1][1], particles.C(parid)[2][1],
              particles.C(parid)[0][2], particles.C(parid)[1][2], particles.C(parid)[2][2]};

        contrib[0] = ((C[0] + C[0]) * model.viscosity - pressure) * vol;
        contrib[1] = (C[1] + C[3]) * model.viscosity * vol;
        contrib[2] = (C[2] + C[6]) * model.viscosity * vol;

        contrib[3] = (C[3] + C[1]) * model.viscosity * vol;
        contrib[4] = ((C[4] + C[4]) * model.viscosity - pressure) * vol;
        contrib[5] = (C[5] + C[7]) * model.viscosity * vol;

        contrib[6] = (C[6] + C[2]) * model.viscosity * vol;
        contrib[7] = (C[7] + C[5]) * model.viscosity * vol;
        contrib[8] = ((C[8] + C[8]) * model.viscosity - pressure) * vol;

        contrib = (C * mass - contrib * dt) * D_inv;
        ivec3 global_base_index{lower_trunc(local_pos[0] * dx_inv + 0.5) - 1,
                                lower_trunc(local_pos[1] * dx_inv + 0.5) - 1,
                                lower_trunc(local_pos[2] * dx_inv + 0.5) - 1};
        local_pos = local_pos - global_base_index * dx;

        vec3x3 ws;
        for (char dd = 0; dd < 3; ++dd) {
          float d = local_pos[dd] * dx_inv - (lower_trunc(local_pos[dd] * dx_inv + 0.5) - 1);
          ws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
          d -= 1.0f;
          ws(dd, 1) = 0.75 - d * d;
          d = 0.5f + d;
          ws(dd, 2) = 0.5 * d * d;
        }

        // float weightsum = 0.f;
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
              using VT = std::decay_t<decltype(
                  std::declval<typename gridblock_t::value_type>().asFloat())>;
              ivec3 offset{i, j, k};
              vec3 xixp = offset * dx - local_pos;
              float W = ws(0, i) * ws(1, j) * ws(2, k);
              // weightsum += W;
              VT wm = mass * W;
              ivec3 local_index = global_base_index + offset;

#if 0
              int blockno = partition.query(local_index / gridblock_t::side_length);
#else
              ivec3 block_coord = local_index;
              for (int d = 0; d < particles_t::dim; ++d)
                block_coord[d] += (local_index[d] < 0 ? -gridblock_t::side_length + 1 : 0);
              block_coord = block_coord / gridblock_t::side_length;
              int blockno = partition.query(block_coord);
#endif

              auto& grid_block = gridblocks[blockno];
#if 0
              for (int d = 0; d < 3; ++d) local_index[d] %= gridblock_t::side_length;
#else
              local_index = local_index - block_coord * gridblock_t::side_length;
#endif

              atomicAdd(&grid_block(0, local_index).asFloat(), wm);
              atomicAdd(
                  &grid_block(1, local_index).asFloat(),
                  (VT)(wm * vel[0]
                       + (contrib[0] * xixp[0] + contrib[3] * xixp[1] + contrib[6] * xixp[2]) * W));
              atomicAdd(
                  &grid_block(2, local_index).asFloat(),
                  (VT)(wm * vel[1]
                       + (contrib[1] * xixp[0] + contrib[4] * xixp[1] + contrib[7] * xixp[2]) * W));
              atomicAdd(
                  &grid_block(3, local_index).asFloat(),
                  (VT)(wm * vel[2]
                       + (contrib[2] * xixp[0] + contrib[5] * xixp[1] + contrib[8] * xixp[2]) * W));
            }
      }
    }

    model_t model;
    particles_t particles;
    partition_t partition;
    gridblocks_t gridblocks;
    float dt;
  };

}  // namespace zs