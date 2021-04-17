#pragma once
#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  template <typename T, typename Table, typename Position> struct ComputeSparsity;

  template <execspace_e space, typename T, typename Table, typename X>
  ComputeSparsity(wrapv<space>, T, int, Table, X)
      -> ComputeSparsity<T, HashTableProxy<space, Table>, VectorProxy<space, X>>;

  enum struct p2g_e : char { apic = 0, flip };
  template <p2g_e, typename T, typename TableT, typename ParticlesT, typename GridBlocksT>
  struct P2GTransfer;

}  // namespace zs