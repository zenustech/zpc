#pragma once
#include "../transfer/Scheme.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/container/Structure.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Collider.h"

namespace zs {

  template <typename GridBlocksT> struct CleanGridBlocks;
  template <typename TableT, typename GridBlocksT> struct PrintGridBlocks;
  template <transfer_scheme_e, typename GridBlocksT> struct ComputeGridBlockVelocity;
  template <typename ColliderT, typename TableT, typename GridBlocksT>
  struct ApplyBoundaryConditionOnGridBlocks;

  template <execspace_e space, typename GridBlocksT> CleanGridBlocks(wrapv<space>, GridBlocksT)
      -> CleanGridBlocks<GridBlocksProxy<space, GridBlocksT>>;
  template <execspace_e space, typename TableT, typename GridBlocksT>
  PrintGridBlocks(wrapv<space>, TableT, GridBlocksT)
      -> PrintGridBlocks<HashTableProxy<space, TableT>, GridBlocksProxy<space, GridBlocksT>>;
  template <execspace_e space, transfer_scheme_e scheme, typename GridBlocksT>
  ComputeGridBlockVelocity(wrapv<space>, wrapv<scheme>, GridBlocksT, float dt, float gravity,
                           float* maxVel)
      -> ComputeGridBlockVelocity<scheme, GridBlocksProxy<space, GridBlocksT>>;
  template <execspace_e space, typename LevelsetT, typename TableT, typename GridBlocksT>
  ApplyBoundaryConditionOnGridBlocks(wrapv<space>, Collider<LevelsetT>, TableT, GridBlocksT)
      -> ApplyBoundaryConditionOnGridBlocks<Collider<LevelsetT>, HashTableProxy<space, TableT>,
                                            GridBlocksProxy<space, GridBlocksT>>;

}  // namespace zs