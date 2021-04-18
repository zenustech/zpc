#pragma once
#include "zensim/simulation/gridop/GridOp.hpp"

namespace zs {

  template <transfer_scheme_e scheme, typename GridBlocksT>
  struct ComputeGridBlockVelocity<scheme, GridBlocksProxy<execspace_e::cuda, GridBlocksT>> {
    using gridblocks_t = GridBlocksProxy<execspace_e::cuda, GridBlocksT>;
    using gridblock_t = typename gridblocks_t::block_t;

    explicit ComputeGridBlockVelocity(wrapv<execspace_e::cuda>, wrapv<scheme>,
                                      GridBlocksT &gridblocks, float dt, float gravity)
        : gridblocks{proxy<execspace_e::cuda>(gridblocks)}, dt{dt}, gravity{gravity} {}

    __forceinline__ __device__ void operator()(typename gridblocks_t::size_type blockid,
                                               typename gridblock_t::size_type cellid) noexcept {
      auto &block = gridblocks[blockid];
      using VT = std::decay_t<decltype(std::declval<typename gridblock_t::value_type>().asFloat())>;
      VT mass = block(0, cellid);
      if (mass > (VT)0) {
        mass = (VT)1 / mass;
        vec<VT, 3> vel;
        for (int d = 0; d < gridblocks_t::dim; ++d) {
          vel[d] = block(d + 1, cellid).asFloat() * mass;
        }
        vel[1] += gravity * dt;
      }
    }

    gridblocks_t gridblocks;
    float dt;
    float gravity;
  };

}  // namespace zs