#include "HashTable.hpp"

namespace zs {

#define INSTANTIATE_HASHTABLE(CoordIndexType, IndexType)                         \
  template struct HashTable<CoordIndexType, 1, IndexType, ZSPmrAllocator<>>;     \
  template struct HashTable<CoordIndexType, 2, IndexType, ZSPmrAllocator<>>;     \
  template struct HashTable<CoordIndexType, 3, IndexType, ZSPmrAllocator<>>;     \
  template struct HashTable<CoordIndexType, 4, IndexType, ZSPmrAllocator<>>;     \
  template struct HashTable<CoordIndexType, 1, IndexType, ZSPmrAllocator<true>>; \
  template struct HashTable<CoordIndexType, 2, IndexType, ZSPmrAllocator<true>>; \
  template struct HashTable<CoordIndexType, 3, IndexType, ZSPmrAllocator<true>>; \
  template struct HashTable<CoordIndexType, 4, IndexType, ZSPmrAllocator<true>>;

  INSTANTIATE_HASHTABLE(i32, i32)
  INSTANTIATE_HASHTABLE(i32, i64)
  INSTANTIATE_HASHTABLE(i64, i64)

}  // namespace zs