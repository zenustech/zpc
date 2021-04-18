#pragma once
#include "zensim/cuda/DeviceUtils.cuh"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/physics/ConstitutiveModel.hpp"
#include "zensim/simulation/transfer/G2P.hpp"

namespace zs {

  template <transfer_scheme_e scheme, typename ModelT, typename GridBlocksT, typename TableT,
            typename ParticlesT>
  struct G2PTransfer<scheme, ModelT, GridBlocksProxy<execspace_e::cuda, GridBlocksT>,
                     HashTableProxy<execspace_e::cuda, TableT>,
                     ParticlesProxy<execspace_e::cuda, ParticlesT>> {
    using model_t = ModelT;  ///< constitutive model
    using gridblocks_t = GridBlocksProxy<execspace_e::cuda, GridBlocksT>;
    using gridblock_t = typename gridblocks_t::block_t;
    using partition_t = HashTableProxy<execspace_e::cuda, TableT>;
    using particles_t = ParticlesProxy<execspace_e::cuda, ParticlesT>;
    static_assert(particles_t::dim == partition_t::dim && particles_t::dim == gridblocks_t::dim,
                  "[particle-partition-grid] dimension mismatch");

    explicit G2PTransfer(wrapv<execspace_e::cuda>, wrapv<scheme>, float dt, const ModelT& model,
                         GridBlocksT& gridblocks, TableT& table, ParticlesT& particles)
        : model{model},
          gridblocks{proxy<execspace_e::cuda>(gridblocks)},
          partition{proxy<execspace_e::cuda>(table)},
          particles{proxy<execspace_e::cuda>(particles)},
          dt{dt} {}

    constexpr float dxinv() const {
      return static_cast<decltype(gridblocks._dx.asFloat())>(1.0) / gridblocks._dx.asFloat();
    }

    __forceinline__ __device__ void operator()(typename particles_t::size_type parid) noexcept {
      if constexpr (particles_t::dim == 3 && std::is_same_v<model_t, EquationOfStateConfig>) {
        float const dx = gridblocks._dx.asFloat();
        float const dx_inv = dxinv();
        float const D_inv = 4.f * dx_inv * dx_inv;
        using ivec3 = vec<int, 3>;
        using vec3 = vec<float, 3>;
        using vec9 = vec<float, 9>;
        using vec3x3 = vec<float, 3, 3>;
        vec3 pos{particles.pos(parid)[0], particles.pos(parid)[1], particles.pos(parid)[2]};
        vec3 vel{vec3::zeros()};

        ivec3 global_base_index{};
        for (int d = 0; d < 3; ++d) global_base_index[d] = (int)std::lround(pos[d] * dx_inv) - 1;
        ivec3 local_base_index = global_base_index;
        vec3 local_pos = pos - local_base_index * dx;

        vec3x3 ws;
#pragma unroll 3
        for (int dd = 0; dd < 3; ++dd) {
          float d = local_pos[dd] * dx_inv - ((int)std::lround(local_pos[dd] * dx_inv) - 1);
          ws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
          d -= 1.0f;
          ws(dd, 1) = 0.75 - d * d;
          d = 0.5f + d;
          ws(dd, 2) = 0.5 * d * d;
          local_base_index[dd] = ((local_base_index[dd] - 1) % gridblock_t::side_length) + 1;
        }

        vec9 C{vec9::zeros()};
#pragma unroll 3
        for (char i = 0; i < 3; i++)
#pragma unroll 3
          for (char j = 0; j < 3; j++)
            for (char k = 0; k < 3; k++) {
              ivec3 offset{i, j, k};
              vec3 xixp = offset * dx - local_pos;
              ivec3 local_index = global_base_index + offset;
              float W = ws(0, i) * ws(1, j) * ws(2, k);
              int blockno = partition.query(ivec3{local_index[0] / gridblock_t::side_length,
                                                  local_index[1] / gridblock_t::side_length,
                                                  local_index[2] / gridblock_t::side_length});
              auto& grid_block = gridblocks[blockno];
              for (int d = 0; d < 3; ++d) local_index[d] %= gridblock_t::side_length;
              vec3 vi{grid_block(1, local_index).asFloat(), grid_block(2, local_index).asFloat(),
                      grid_block(3, local_index).asFloat()};

              vel += vi * W;
              for (int i = 0; i < 9; ++i) C[i] += W * vi(i % 3) * xixp(i / 3);
            }
        pos += vel * dt;

        float J = particles.J(parid);
        J = (1 + (C[0] + C[4] + C[8]) * dt * D_inv) * J;
        if (J < 0.1) J = 0.1;
        particles.J(parid) = J;
        for (int i = 0; i < 3; ++i) particles.pos(parid)[i] = pos[i];
        for (int i = 0; i < 3; ++i) particles.vel(parid)[i] = vel[i];
        for (int i = 0; i < 9; ++i) particles.C(parid)[i % 3][i / 3] = C[i];
      }
    }

    model_t model;
    gridblocks_t gridblocks;
    partition_t partition;
    particles_t particles;
    float dt;
  };

}  // namespace zs