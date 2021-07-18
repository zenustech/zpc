#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structure.hpp"
#include "zensim/types/Tuple.h"

__device__ int sqr(int n) { return n * n; }

extern "C" __global__ void zpc_prebuilt_kernel() {
  auto id = zs::make_tuple(blockIdx.x, threadIdx.x);
  printf("what the heck, indeed called successfully at (%d, %d), result: %d!\n",
         (int)id.template get<0>(), (int)id.template get<1>(), sqr(threadIdx.x));
}

#if 0
__device__ void check_grid_channels(float *v);

namespace zs {

  template <typename GridBlocksT>
  __global__ void inspect_grid(GridBlocksProxy<execspace_e::cuda, GridBlocksT> gridblocks) {
    using gridblocks_t = GridBlocksProxy<execspace_e::cuda, GridBlocksT>;
    using gridblock_t = typename gridblocks_t::block_t;
    typename gridblock_t::size_type blockid = blockIdx.x;
    typename gridblock_t::size_type cellid = threadIdx.x;

    auto &block = gridblocks[blockid];
    using VT = std::decay_t<decltype(std::declval<typename gridblock_t::value_type>().asFloat())>;
    VT mass = block(0, cellid).asFloat();
    check_grid_channels(&mass);
  }

}  // namespace zs
// GridBlocks<GridBlock<dat32, 3, 2, 2>>
extern "C" __global__ void inspect_grid();
#endif