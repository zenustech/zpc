#pragma once
#include "zensim/cuda/DeviceUtils.cuh"
#include "zensim/simulation/gridop/GridOp.hpp"

namespace zs {

  template <typename GridBlocksT>
  struct CleanGridBlocks<GridBlocksProxy<execspace_e::cuda, GridBlocksT>> {
    using gridblocks_t = GridBlocksProxy<execspace_e::cuda, GridBlocksT>;
    using gridblock_t = typename gridblocks_t::block_t;

    explicit CleanGridBlocks(wrapv<execspace_e::cuda>, GridBlocksT &gridblocks)
        : gridblocks{proxy<execspace_e::cuda>(gridblocks)} {}

    __forceinline__ __device__ void operator()(typename gridblocks_t::size_type blockid,
                                               typename gridblock_t::size_type cellid) noexcept {
      auto &block = gridblocks[blockid];
      using VT = std::decay_t<decltype(std::declval<typename gridblock_t::value_type>().asFloat())>;
      block(0, cellid).asFloat() = static_cast<VT>(0);
      for (int d = 0; d < gridblocks_t::dim; ++d)
        block(d + 1, cellid).asFloat() = static_cast<VT>(0);
    }

    gridblocks_t gridblocks;
  };

  template <typename GridBlocksT>
  struct PrintGridBlocks<GridBlocksProxy<execspace_e::cuda, GridBlocksT>> {
    using gridblocks_t = GridBlocksProxy<execspace_e::cuda, GridBlocksT>;
    using gridblock_t = typename gridblocks_t::block_t;

    explicit PrintGridBlocks(wrapv<execspace_e::cuda>, GridBlocksT &gridblocks)
        : gridblocks{proxy<execspace_e::cuda>(gridblocks)} {}

    __forceinline__ __device__ void operator()(typename gridblocks_t::size_type blockid,
                                               typename gridblock_t::size_type cellid) noexcept {
      auto &block = gridblocks[blockid];
      using VT = std::decay_t<decltype(std::declval<typename gridblock_t::value_type>().asFloat())>;
      if (blockid == 0 && cellid == 0) printf("block space: %d\n", gridblock_t::space);
      if (blockid == 0)  // || (float)block(1, cellid).asFloat() > 0)
        printf("(%d, %d): mass %e, (%e, %e, %e)\n", (int)blockid, (int)cellid,
               block(0, cellid).asFloat(), block(1, cellid).asFloat(), block(2, cellid).asFloat(),
               block(3, cellid).asFloat());
    }

    gridblocks_t gridblocks;
  };

  template <transfer_scheme_e scheme, typename GridBlocksT>
  struct ComputeGridBlockVelocity<scheme, GridBlocksProxy<execspace_e::cuda, GridBlocksT>> {
    using gridblocks_t = GridBlocksProxy<execspace_e::cuda, GridBlocksT>;
    using gridblock_t = typename gridblocks_t::block_t;

    explicit ComputeGridBlockVelocity(wrapv<execspace_e::cuda>, wrapv<scheme>,
                                      GridBlocksT &gridblocks, float dt, float gravity,
                                      float *maxVel)
        : gridblocks{proxy<execspace_e::cuda>(gridblocks)},
          dt{dt},
          gravity{gravity},
          maxVel{maxVel} {}

    __forceinline__ __device__ void operator()(typename gridblocks_t::size_type blockid,
                                               typename gridblock_t::size_type cellid) noexcept {
      auto &block = gridblocks[blockid];
      using VT = std::decay_t<decltype(std::declval<typename gridblock_t::value_type>().asFloat())>;
      VT mass = block(0, cellid).asFloat();
      if (mass > (VT)0) {
        mass = (VT)1 / mass;
        vec<VT, gridblocks_t::dim> vel;
        for (int d = 0; d < gridblocks_t::dim; ++d) {
          vel[d] = block(d + 1, cellid).asFloat() * mass;
        }
        vel[1] += gravity * dt;

#if 0
        if (blockid < 1)
          printf("block %d, cell %d, mass %e, (%e, %e, %e) ref:(%e, %e, %e)\n", (int)blockid,
                 (int)cellid, block(0, cellid).asFloat(), (float)vel[0], (float)vel[1],
                 (float)vel[2], (float)block(1, cellid).asFloat(),
                 (float)block(2, cellid).asFloat(), (float)block(3, cellid).asFloat());
#endif

        /// write back
        for (int d = 0; d < gridblocks_t::dim; ++d) block(d + 1, cellid).asFloat() = vel[d];

        /// cfl dt
        float ret{0.f};
        for (int d = 0; d < gridblocks_t::dim; ++d) ret += vel[d] * vel[d];
        atomicMax<float>(maxVel, ret);
      }
    }

    gridblocks_t gridblocks;
    float dt;
    float gravity;
    float *maxVel;
  };

}  // namespace zs