#include "AdaptiveGrid.hpp"

namespace zs {

  // openvdb float grid
  ZPC_INSTANTIATE_STRUCT
  AdaptiveGridImpl<3, f32, index_sequence<3, 4, 5>, index_sequence<3, 4, 5>,
                   index_sequence<0, 1, 2>, ZSPmrAllocator<>>;
  ZPC_INSTANTIATE_STRUCT VdbGrid<3, f32, index_sequence<3, 4, 5>, ZSPmrAllocator<>>;
  // bifrost adaptive tile tree
  ZPC_INSTANTIATE_STRUCT
  AdaptiveGridImpl<3, f32, index_sequence<2, 2, 2>, index_sequence<1, 1, 1>,
                   index_sequence<0, 1, 2>, ZSPmrAllocator<>>;
  ZPC_INSTANTIATE_STRUCT AdaptiveTileTree<3, f32, 3, 2, ZSPmrAllocator<>>;

}  // namespace zs