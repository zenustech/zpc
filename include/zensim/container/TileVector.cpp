#include "TileVector.hpp"

namespace zs {

#define INSTANTIATE_TILEVECTOR(LENGTH)                                          \
  template struct TileVector<u32, LENGTH, unsigned char, ZSPmrAllocator<>>;     \
  template struct TileVector<u64, LENGTH, unsigned char, ZSPmrAllocator<>>;     \
  template struct TileVector<i32, LENGTH, unsigned char, ZSPmrAllocator<>>;     \
  template struct TileVector<i64, LENGTH, unsigned char, ZSPmrAllocator<>>;     \
  template struct TileVector<f32, LENGTH, unsigned char, ZSPmrAllocator<>>;     \
  template struct TileVector<f64, LENGTH, unsigned char, ZSPmrAllocator<>>;     \
  template struct TileVector<u32, LENGTH, unsigned char, ZSPmrAllocator<true>>; \
  template struct TileVector<u64, LENGTH, unsigned char, ZSPmrAllocator<true>>; \
  template struct TileVector<i32, LENGTH, unsigned char, ZSPmrAllocator<true>>; \
  template struct TileVector<i64, LENGTH, unsigned char, ZSPmrAllocator<true>>; \
  template struct TileVector<f32, LENGTH, unsigned char, ZSPmrAllocator<true>>; \
  template struct TileVector<f64, LENGTH, unsigned char, ZSPmrAllocator<true>>;

  /// 8, 32, 64, 512
  INSTANTIATE_TILEVECTOR(8)
  INSTANTIATE_TILEVECTOR(32)
  INSTANTIATE_TILEVECTOR(64)
  INSTANTIATE_TILEVECTOR(512)

}  // namespace zs