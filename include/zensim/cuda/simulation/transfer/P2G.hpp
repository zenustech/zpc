#pragma once
#include "zensim/cuda/DeviceUtils.cuh"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/physics/ConstitutiveModel.hpp"
#include "zensim/simulation/transfer/P2G.hpp"

namespace zs {

  template <typename Table> struct CleanSparsity<HashTableProxy<execspace_e::cuda, Table>> {
    using table_t = HashTableProxy<execspace_e::cuda, Table>;

    explicit CleanSparsity(wrapv<execspace_e::cuda>, Table& table)
        : table{proxy<execspace_e::cuda>(table)} {}

    __forceinline__ __device__ void operator()(typename Table::value_t entry) noexcept {
      using namespace placeholders;
      table._table(_0, entry) = Table::key_t::uniform(Table::key_scalar_sentinel_v);
      table._table(_2, entry) = -1;
      if (entry == 0) *table._cnt = 0;
    }

    table_t table;
  };

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

  template <typename Table> struct EnlargeSparsity<HashTableProxy<execspace_e::cuda, Table>> {
    static constexpr int dim = Table::dim;
    using table_t = HashTableProxy<execspace_e::cuda, Table>;

    explicit EnlargeSparsity(wrapv<execspace_e::cuda>, Table& table, vec<int, dim> lo,
                             vec<int, dim> hi)
        : table{proxy<execspace_e::cuda>(table)}, lo{lo}, hi{hi} {}

    __forceinline__ __device__ void operator()(typename table_t::value_t i) noexcept {
      if constexpr (table_t::dim == 3) {
        auto blockid = table._activeKeys[i];
        for (int dx = lo[0]; dx < hi[0]; ++dx)
          for (int dy = lo[1]; dy < hi[1]; ++dy)
            for (int dz = lo[2]; dz < hi[2]; ++dz) {
              table.insert(blockid + vec<int, 3>{dx, dy, dz});
              // printf("Wow, find neighbors (%d, %d, %d)!\n", dx, dy, dz);
            }
      } else if constexpr (table_t::dim == 2) {
        auto blockid = table._activeKeys[i];
        for (int dx = lo[0]; dx < hi[0]; ++dx)
          for (int dy = lo[1]; dy < hi[1]; ++dy) table.insert(blockid + vec<int, 2>{dx, dy});
      }
    }

    table_t table;
    vec<int, dim> lo, hi;
  };

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
      if constexpr (particles_t::dim == 3 && std::is_same_v<model_t, EquationOfStateConfig>) {
        float const dx = gridblocks._dx.asFloat();
        float const dx_inv = dxinv();
        float const D_inv = 4.f * dx_inv * dx_inv;
        using ivec3 = vec<int, 3>;
        using vec3 = vec<float, 3>;
        using vec9 = vec<float, 9>;
        vec3 local_pos{particles.pos(parid)[0], particles.pos(parid)[1], particles.pos(parid)[2]};
        vec3 vel{particles.vel(parid)[0], particles.vel(parid)[1], particles.vel(parid)[2]};
        float J = particles.J(parid);
        float mass = particles.mass(parid);
        float vol = model.volume * J;  ///< this is hack!

        float pressure = model.bulk * (powf(J, -model.gamma) - 1.f);

        vec9 contrib,
            C{particles.C(parid)[0][0], particles.C(parid)[1][0], particles.C(parid)[2][0],
              particles.C(parid)[0][1], particles.C(parid)[1][1], particles.C(parid)[2][1],
              particles.C(parid)[0][2], particles.C(parid)[1][2], particles.C(parid)[2][2]};
        contrib[0] = ((C[0] + C[0]) * D_inv * model.viscosity - pressure) * vol;
        contrib[1] = (C[1] + C[3]) * D_inv * model.viscosity * vol;
        contrib[2] = (C[2] + C[6]) * D_inv * model.viscosity * vol;

        contrib[3] = (C[3] + C[1]) * D_inv * model.viscosity * vol;
        contrib[4] = ((C[4] + C[4]) * D_inv * model.viscosity - pressure) * vol;
        contrib[5] = (C[5] + C[7]) * D_inv * model.viscosity * vol;

        contrib[6] = (C[6] + C[2]) * D_inv * model.viscosity * vol;
        contrib[7] = (C[7] + C[5]) * D_inv * model.viscosity * vol;
        contrib[8] = ((C[8] + C[8]) * D_inv * model.viscosity - pressure) * vol;

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
              using VT = std::decay_t<decltype(
                  std::declval<typename gridblock_t::value_type>().asFloat())>;
              ivec3 offset{i, j, k};
              vec3 xixp = offset * dx - local_pos;
              VT W = dws[0][i] * dws[1][j] * dws[2][k];
              ivec3 local_index = global_base_index + offset;
              VT wm = mass * W;
              int blockno = partition.query(ivec3{local_index[0] / gridblock_t::side_length,
                                                  local_index[1] / gridblock_t::side_length,
                                                  local_index[2] / gridblock_t::side_length});
              auto& grid_block = gridblocks[blockno];
              for (int d = 0; d < 3; ++d) local_index[d] %= gridblock_t::side_length;
              atomicAddFloat(&grid_block(0, local_index).asFloat(), wm);
              atomicAddFloat(
                  &grid_block(1, local_index).asFloat(),
                  (VT)(wm * vel[0]
                       + (contrib[0] * xixp[0] + contrib[3] * xixp[1] + contrib[6] * xixp[2]) * W));
              atomicAddFloat(
                  &grid_block(2, local_index).asFloat(),
                  (VT)(wm * vel[1]
                       + (contrib[1] * xixp[0] + contrib[4] * xixp[1] + contrib[7] * xixp[2]) * W));
              atomicAddFloat(
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