#pragma once
#include "Scheme.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/container/Structure.hpp"
#include "zensim/container/Structurefree.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  template <typename ParticlesT> struct SetParticleAttribute;
  template <typename Table> struct CleanSparsity;
  template <typename T, typename Table, typename Position> struct ComputeSparsity;
  template <typename Table> struct EnlargeSparsity;
  template <transfer_scheme_e, typename ConstitutiveModel, typename ParticlesT, typename TableT,
            typename GridBlocksT>
  struct P2GTransfer;

  template <execspace_e space, typename ParticlesT> SetParticleAttribute(wrapv<space>, ParticlesT)
      -> SetParticleAttribute<ParticlesProxy<space, ParticlesT>>;

  template <execspace_e space, typename Table> CleanSparsity(wrapv<space>, Table)
      -> CleanSparsity<HashTableProxy<space, Table>>;

  template <execspace_e space, typename T, typename Table, typename X>
  ComputeSparsity(wrapv<space>, T, int, Table, X)
      -> ComputeSparsity<T, HashTableProxy<space, Table>, VectorProxy<space, X>>;

  template <execspace_e space, typename Table>
  EnlargeSparsity(wrapv<space>, Table, vec<int, Table::dim>, vec<int, Table::dim>)
      -> EnlargeSparsity<HashTableProxy<space, Table>>;

  template <execspace_e space, transfer_scheme_e scheme, typename Model, typename ParticlesT,
            typename TableT, typename GridBlocksT>
  P2GTransfer(wrapv<space>, wrapv<scheme>, float, Model, ParticlesT, TableT, GridBlocksT)
      -> P2GTransfer<scheme, Model, ParticlesProxy<space, ParticlesT>,
                     HashTableProxy<space, TableT>, GridBlocksProxy<space, GridBlocksT>>;

}  // namespace zs