#pragma once
#include "zensim/container/HashTable.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/simulation/init/Structure.hpp"
#include "zensim/simulation/init/Structurefree.hpp"

namespace zs {

  enum struct p2g_e : char { apic = 0, flip };
  template <typename Table> struct CleanSparsity;
  template <typename T, typename Table, typename Position> struct ComputeSparsity;
  template <typename Table> struct EnlargeSparsity;
  template <p2g_e, typename ConstitutiveModel, typename ParticlesT, typename TableT,
            typename GridBlocksT>
  struct P2GTransfer;

  template <execspace_e space, typename Table> CleanSparsity(wrapv<space>, Table)
      -> CleanSparsity<HashTableProxy<space, Table>>;
  template <execspace_e space, typename T, typename Table, typename X>
  ComputeSparsity(wrapv<space>, T, int, Table, X)
      -> ComputeSparsity<T, HashTableProxy<space, Table>, VectorProxy<space, X>>;
  template <execspace_e space, typename Table>
  EnlargeSparsity(wrapv<space>, Table, vec<int, Table::dim>, vec<int, Table::dim>)
      -> EnlargeSparsity<HashTableProxy<space, Table>>;

  template <execspace_e space, p2g_e scheme, typename Model, typename ParticlesT, typename TableT,
            typename GridBlocksT>
  P2GTransfer(wrapv<space>, wrapv<scheme>, float, Model, ParticlesT, TableT, GridBlocksT)
      -> P2GTransfer<scheme, Model, ParticlesProxy<space, ParticlesT>,
                     HashTableProxy<space, TableT>, GridBlocksProxy<space, GridBlocksT>>;

}  // namespace zs