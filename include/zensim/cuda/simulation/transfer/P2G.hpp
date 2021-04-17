#pragma once
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/simulation/transfer/P2G.hpp"

namespace zs {

  template <typename T, typename Table, typename X>
  struct ComputeSparsity<T, HashTableProxy<execspace_e::cuda, Table>,
                         VectorProxy<execspace_e::cuda, X>> {
    using table_t = HashTableProxy<execspace_e::cuda, Table>;
    using positions_t = VectorProxy<execspace_e::cuda, X>;

    explicit ComputeSparsity(wrapv<execspace_e::cuda>, T dx, int blockLen, Table& table, X& pos)
        : dxinv{1.0 / dx},
          blockLen{blockLen},
          table{proxy<execspace_e::cuda>(table)},
          pos{proxy<execspace_e::cuda>(pos)} {}

    __forceinline__ __device__ void operator()(typename positions_t::size_type parid) noexcept {
      if constexpr (table_t::dim == 3) {
        auto coord = vec<int, 3>{std::lround(pos(parid)[0] * dxinv) - 2,
                                 std::lround(pos(parid)[1] * dxinv) - 2,
                                 std::lround(pos(parid)[2] * dxinv) - 2};
        auto blockid = coord / blockLen;
        table.insert(blockid);
      } else if constexpr (table_t::dim == 2) {
        auto coord = vec<int, 2>{std::lround(pos(parid)[0] * dxinv) - 2,
                                 std::lround(pos(parid)[1] * dxinv) - 2};
        auto blockid = coord / blockLen;
        table.insert(blockid);
      }
    }

    T dxinv;
    int blockLen;
    table_t table;
    positions_t pos;
  };

#if 0
  template <p2g_e scheme, typename T, typename TableT, typename ParticlesT, typename GridBlocksT>
  struct P2GTransfer<T, HashTableProxy<execspace_e::cuda, Table>,
                     ParticlesProxy<execspace_e::cuda, ParticlesT>,
                     GridBlocksProxy<execspace_e::cuda, GridBlocksT>> {
    using table_t = HashTableProxy<execspace_e::cuda, Table>;
    using positions_t = ParticlesProxy<execspace_e::cuda, Particles>;
    using gridblocks_t = GridBlocksProxy<execspace_e::cuda, GridBlocks>;

    explicit ComputeSparsity(wrapv<execspace_e::cuda>, T dx, Table& table, X& pos)
        : dxinv{1.0 / dx},
          table{proxy<execspace_e::cuda>(table)},
          pos{proxy<execspace_e::cuda>(pos)} {}

    __forceinline__ __device__ void operator()(typename positions_t::size_type parid) noexcept {
      float const D_inv = 4.f * dx_inv * dx_inv;
      uint32_t parid = blockIdx.x * blockDim.x + threadIdx.x;
      if (parid >= particleCount) return;

      vec3 local_pos{parray(0, parid), parray(1, parid), parray(2, parid)};
      vec3 vel = v0;
      /// disable disturbance

      vec9 contrib, C;
      contrib.set(0.f), C.set(0.f);
      contrib = (C * mass - contrib * dt) * D_inv;
      ivec3 global_base_index{int(std::lround(local_pos[0] * dx_inv) - 1),
                              int(std::lround(local_pos[1] * dx_inv) - 1),
                              int(std::lround(local_pos[2] * dx_inv) - 1)};
      local_pos = local_pos - global_base_index * dx;
      vec<vec3, 3> dws;
      for (int d = 0; d < 3; ++d) dws[d] = bspline_weight(local_pos[d], dx_inv);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          for (int k = 0; k < 3; ++k) {
            ivec3 offset{i, j, k};
            vec3 xixp = offset * dx - local_pos;
            float W = dws[0][i] * dws[1][j] * dws[2][k];
            ivec3 local_index = global_base_index + offset;
            float wm = mass * W;
            int blockno = partition.query(ivec3{local_index[0] >> g_blockbits,
                                                local_index[1] >> g_blockbits,
                                                local_index[2] >> g_blockbits});
            auto grid_block = grid(blockno);
            for (int d = 0; d < 3; ++d) local_index[d] &= g_blockmask;
            atomicAdd(&grid_block(0, local_index[0], local_index[1], local_index[2]), wm);
            atomicAdd(
                &grid_block(1, local_index[0], local_index[1], local_index[2]),
                wm * vel[0]
                    + (contrib[0] * xixp[0] + contrib[3] * xixp[1] + contrib[6] * xixp[2]) * W);
            atomicAdd(
                &grid_block(2, local_index[0], local_index[1], local_index[2]),
                wm * vel[1]
                    + (contrib[1] * xixp[0] + contrib[4] * xixp[1] + contrib[7] * xixp[2]) * W);
            atomicAdd(
                &grid_block(3, local_index[0], local_index[1], local_index[2]),
                wm * vel[2]
                    + (contrib[2] * xixp[0] + contrib[5] * xixp[1] + contrib[8] * xixp[2]) * W);
          }

      auto coord = vec<int, 3>{std::lround(pos(parid)[0] * dxinv) - 2,
                               std::lround(pos(parid)[1] * dxinv) - 2,
                               std::lround(pos(parid)[2] * dxinv) - 2};
      auto blockid = coord / blockLen;
      table.insert(blockid);
    }

    float dt;
    partition_t partition;
    particles_t particles;
    grid_t grid;
  };
#endif

}  // namespace zs