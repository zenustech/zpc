#include "Bht.hpp"

namespace zs {

#define INSTANTIATE_BHT(CoordIndexType, IndexType, B)                                \
  ZPC_INSTANTIATE_STRUCT bht<CoordIndexType, 1, IndexType, B, ZSPmrAllocator<>>;     \
  ZPC_INSTANTIATE_STRUCT bht<CoordIndexType, 2, IndexType, B, ZSPmrAllocator<>>;     \
  ZPC_INSTANTIATE_STRUCT bht<CoordIndexType, 3, IndexType, B, ZSPmrAllocator<>>;     \
  ZPC_INSTANTIATE_STRUCT bht<CoordIndexType, 4, IndexType, B, ZSPmrAllocator<>>;     \
  ZPC_INSTANTIATE_STRUCT bht<CoordIndexType, 1, IndexType, B, ZSPmrAllocator<true>>; \
  ZPC_INSTANTIATE_STRUCT bht<CoordIndexType, 2, IndexType, B, ZSPmrAllocator<true>>; \
  ZPC_INSTANTIATE_STRUCT bht<CoordIndexType, 3, IndexType, B, ZSPmrAllocator<true>>; \
  ZPC_INSTANTIATE_STRUCT bht<CoordIndexType, 4, IndexType, B, ZSPmrAllocator<true>>;

  INSTANTIATE_BHT(i32, i32, 16)
  INSTANTIATE_BHT(i32, i64, 16)

}  // namespace zs