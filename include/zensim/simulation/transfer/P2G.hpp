#pragma once
#include "Scheme.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/execution/Atomics.hpp"
#include "zensim/geometry/Structure.hpp"
#include "zensim/geometry/Structurefree.hpp"

namespace zs {

  template <transfer_scheme_e, typename ConstitutiveModel, typename ParticlesT, typename TableT,
            typename GridT>
  struct P2GTransfer;

  template <execspace_e space, transfer_scheme_e scheme, typename Model, typename ParticlesT,
            typename TableT, typename V, int d, int chnbits, int dombits>
  P2GTransfer(wrapv<space>, wrapv<scheme>, float, Model, ParticlesT, TableT,
              GridBlocks<GridBlock<V, d, chnbits, dombits>>)
      -> P2GTransfer<scheme, Model, ParticlesView<space, ParticlesT>, HashTableView<space, TableT>,
                     GridBlocksView<space, GridBlocks<GridBlock<V, d, chnbits, dombits>>>>;

  template <execspace_e space, transfer_scheme_e scheme, typename Model, typename ParticlesT,
            typename TableT, typename T, int d, auto l>
  P2GTransfer(wrapv<space>, wrapv<scheme>, float, Model, ParticlesT, TableT, Grids<T, d, l>)
      -> P2GTransfer<scheme, Model, ParticlesView<space, ParticlesT>, HashTableView<space, TableT>,
                     GridsView<space, Grids<T, d, l>>>;

}  // namespace zs