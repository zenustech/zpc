#pragma once
#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  enum struct p2g_e : char { apic = 0, flip };
  template <typename Table> struct CleanSparsity;
  template <typename T, typename Table, typename Position> struct ComputeSparsity;

  template <execspace_e space, typename Table> CleanSparsity(wrapv<space>, Table)
      -> CleanSparsity<HashTableProxy<space, Table>>;
  template <execspace_e space, typename T, typename Table, typename X>
  ComputeSparsity(wrapv<space>, T, int, Table, X)
      -> ComputeSparsity<T, HashTableProxy<space, Table>, VectorProxy<space, X>>;

  template <p2g_e, typename T, typename TableT, typename ParticlesT, typename GridBlocksT>
  struct P2GTransfer;

}  // namespace zs