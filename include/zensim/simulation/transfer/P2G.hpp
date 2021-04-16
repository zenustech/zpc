#pragma once
#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  template <typename T, typename Table, typename Position> struct ComputeSparsity;

  template <execspace_e space, typename T, typename Table, typename X>
  ComputeSparsity(wrapv<space>, T, Table, X)
      -> ComputeSparsity<T, HashTableProxy<space, Table>, VectorProxy<space, X>>;

}  // namespace zs