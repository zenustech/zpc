#pragma once
#include "Scheme.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/container/Structure.hpp"
#include "zensim/container/Structurefree.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  template <transfer_scheme_e, typename ConstitutiveModel, typename GridBlocksT, typename TableT,
            typename ParticlesT>
  struct G2PTransfer;

  template <execspace_e space, transfer_scheme_e scheme, typename Model, typename GridBlocksT,
            typename TableT, typename ParticlesT>
  G2PTransfer(wrapv<space>, wrapv<scheme>, float, Model, GridBlocksT, TableT, ParticlesT)
      -> G2PTransfer<scheme, Model, GridBlocksProxy<space, GridBlocksT>,
                     HashTableProxy<space, TableT>, ParticlesProxy<space, ParticlesT>>;

}  // namespace zs