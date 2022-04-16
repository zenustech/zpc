#include "TileVector.hpp"

namespace zs {

#define INSTANTIATE_TILEVECTOR(LENGTH)                                          \
  template struct TileVector<u32, LENGTH, ZSPmrAllocator<>>;     \
  template struct TileVector<u64, LENGTH, ZSPmrAllocator<>>;     \
  template struct TileVector<i32, LENGTH, ZSPmrAllocator<>>;     \
  template struct TileVector<i64, LENGTH, ZSPmrAllocator<>>;     \
  template struct TileVector<f32, LENGTH, ZSPmrAllocator<>>;     \
  template struct TileVector<f64, LENGTH, ZSPmrAllocator<>>;     \
  template struct TileVector<u32, LENGTH, ZSPmrAllocator<true>>; \
  template struct TileVector<u64, LENGTH, ZSPmrAllocator<true>>; \
  template struct TileVector<i32, LENGTH, ZSPmrAllocator<true>>; \
  template struct TileVector<i64, LENGTH, ZSPmrAllocator<true>>; \
  template struct TileVector<f32, LENGTH, ZSPmrAllocator<true>>; \
  template struct TileVector<f64, LENGTH, ZSPmrAllocator<true>>;

  /// 8, 32, 64, 512
  INSTANTIATE_TILEVECTOR(8)
  INSTANTIATE_TILEVECTOR(32)
  INSTANTIATE_TILEVECTOR(64)
  INSTANTIATE_TILEVECTOR(512)

}  // namespace zs