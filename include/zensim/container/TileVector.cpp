#include "TileVector.hpp"

namespace zs {

#define INSTANTIATE_TILEVECTOR(LENGTH)                                  \
  ZPC_INSTANTIATE_STRUCT TileVector<u32, LENGTH, ZSPmrAllocator<>>;     \
  ZPC_INSTANTIATE_STRUCT TileVector<u64, LENGTH, ZSPmrAllocator<>>;     \
  ZPC_INSTANTIATE_STRUCT TileVector<i32, LENGTH, ZSPmrAllocator<>>;     \
  ZPC_INSTANTIATE_STRUCT TileVector<i64, LENGTH, ZSPmrAllocator<>>;     \
  ZPC_INSTANTIATE_STRUCT TileVector<f32, LENGTH, ZSPmrAllocator<>>;     \
  ZPC_INSTANTIATE_STRUCT TileVector<f64, LENGTH, ZSPmrAllocator<>>;     \
  ZPC_INSTANTIATE_STRUCT TileVector<u32, LENGTH, ZSPmrAllocator<true>>; \
  ZPC_INSTANTIATE_STRUCT TileVector<u64, LENGTH, ZSPmrAllocator<true>>; \
  ZPC_INSTANTIATE_STRUCT TileVector<i32, LENGTH, ZSPmrAllocator<true>>; \
  ZPC_INSTANTIATE_STRUCT TileVector<i64, LENGTH, ZSPmrAllocator<true>>; \
  ZPC_INSTANTIATE_STRUCT TileVector<f32, LENGTH, ZSPmrAllocator<true>>; \
  ZPC_INSTANTIATE_STRUCT TileVector<f64, LENGTH, ZSPmrAllocator<true>>;

  /// 8, 32, 64, 512
  INSTANTIATE_TILEVECTOR(8)
  INSTANTIATE_TILEVECTOR(32)
  INSTANTIATE_TILEVECTOR(64)
  INSTANTIATE_TILEVECTOR(512)

}  // namespace zs