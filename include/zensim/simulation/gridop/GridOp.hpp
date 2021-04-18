#pragma once
#include "../transfer/Scheme.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/simulation/init/Structure.hpp"

namespace zs {

  template <transfer_scheme_e, typename GridBlocksT> struct ComputeGridBlockVelocity;

  template <execspace_e space, transfer_scheme_e scheme, typename GridBlocksT>
  ComputeGridBlockVelocity(wrapv<space>, wrapv<scheme>, GridBlocksT, float dt, float gravity)
      -> ComputeGridBlockVelocity<scheme, GridBlocksProxy<space, GridBlocksT>>;

}  // namespace zs