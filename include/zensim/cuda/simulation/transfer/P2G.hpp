#pragma once
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/simulation/transfer/P2G.hpp"

namespace zs {

  template <typename T, typename Table, typename X>
  struct ComputeSparsity<T, HashTableProxy<execspace_e::cuda, Table>,
                         VectorProxy<execspace_e::cuda, X>> {
    using table_t = HashTableProxy<execspace_e::cuda, Table>;
    using positions_t = VectorProxy<execspace_e::cuda, X>;

    explicit ComputeSparsity(wrapv<execspace_e::cuda>, T dx, int blockLen, Table& table, X& pos)
        : dxinv{1.0 / dx},
          table{proxy<execspace_e::cuda>(table)},
          pos{proxy<execspace_e::cuda>(pos)} {}

    __forceinline__ __device__ void operator()(typename positions_t::size_type parid) noexcept {
      auto coord = vec<int, 3>{std::lround(pos(parid)[0] * dxinv) - 2,
                               std::lround(pos(parid)[1] * dxinv) - 2,
                               std::lround(pos(parid)[2] * dxinv) - 2};
      auto blockid = coord / blockLen;
      table.insert(blockid);
    }

    T dxinv;
    int blockLen;
    table_t table;
    positions_t pos;
  };

}  // namespace zs