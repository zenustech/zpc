#pragma once
#include "AdaptiveGrid.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  namespace detail {}

  template <typename ExecPol, int dim, typename ValueSrc, size_t... TileBitsSrc,
            size_t... ScalingBitsSrc, typename ValueDst, size_t... TileBitsDst,
            size_t... ScalingBitsDst, size_t... Is>
  void restructure_adaptive_grid(
      ExecPol &&pol,
      const AdaptiveGridImpl<dim, ValueSrc, index_sequence<TileBitsSrc...>,
                             index_sequence<ScalingBitsSrc...>, index_sequence<Is...>,
                             ZSPmrAllocator<>> &agSrc,
      AdaptiveGridImpl<dim, ValueDst, index_sequence<TileBitsDst...>,
                       index_sequence<ScalingBitsDst...>, index_sequence<Is...>, ZSPmrAllocator<>>
          &agDst) {
    constexpr execspace_e space = RM_CVREF_T(pol)::exec_tag::value;

    if (!valid_memspace_for_execution(pol, agSrc.get_allocator())
        || !valid_memspace_for_execution(pol, agDst.get_allocator()))
      throw std::runtime_error(
          "[restructure_adaptive_grid] current memory location not compatible with the execution "
          "policy");

    throw std::runtime_error("[restructure_adaptive_grid] not implemented yet!");

    auto allocator = get_temporary_memory_source(pol);
  }

}  // namespace zs