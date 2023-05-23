#include "zensim/container/TileVector.hpp"
#include "zensim/py_interop/TileVectorView.hpp"

namespace zs {

#define INSTANTIATE_TILE_VECTOR_VIEWLITE(T, L)                                                   \
  TileVectorViewLite<T, L> pyview(TileVector<T, L, ZSPmrAllocator<false>> &v) {                  \
    return TileVectorViewLite<T, L>{v.data(), v.numChannels()};                                  \
  }                                                                                              \
  TileVectorViewLite<const T, L> pyview(const TileVector<T, L, ZSPmrAllocator<false>> &v) {      \
    return TileVectorViewLite<const T, L>{v.data(), v.numChannels()};                            \
  }                                                                                              \
  TileVectorViewLite<T, L> pyview(TileVector<T, L, ZSPmrAllocator<true>> &v) {                   \
    return TileVectorViewLite<T, L>{v.data(), v.numChannels()};                                  \
  }                                                                                              \
  TileVectorViewLite<const T, L> pyview(const TileVector<T, L, ZSPmrAllocator<true>> &v) {       \
    return TileVectorViewLite<const T, L>{v.data(), v.numChannels()};                            \
  }                                                                                              \
  TileVectorNamedViewLite<T, L> pyview(true_type, TileVector<T, L, ZSPmrAllocator<false>> &v) {  \
    return TileVectorNamedViewLite<T, L>{v.data(),          v.numChannels(),                     \
                                         v.tagNameHandle(), v.tagOffsetHandle(),                 \
                                         v.tagSizeHandle(), v.numProperties()};                  \
  }                                                                                              \
  TileVectorNamedViewLite<const T, L> pyview(true_type,                                          \
                                             const TileVector<T, L, ZSPmrAllocator<false>> &v) { \
    return TileVectorNamedViewLite<const T, L>{v.data(),          v.numChannels(),               \
                                               v.tagNameHandle(), v.tagOffsetHandle(),           \
                                               v.tagSizeHandle(), v.numProperties()};            \
  }                                                                                              \
  TileVectorNamedViewLite<T, L> pyview(true_type, TileVector<T, L, ZSPmrAllocator<true>> &v) {   \
    return TileVectorNamedViewLite<T, L>{v.data(),          v.numChannels(),                     \
                                         v.tagNameHandle(), v.tagOffsetHandle(),                 \
                                         v.tagSizeHandle(), v.numProperties()};                  \
  }                                                                                              \
  TileVectorNamedViewLite<const T, L> pyview(true_type,                                          \
                                             const TileVector<T, L, ZSPmrAllocator<true>> &v) {  \
    return TileVectorNamedViewLite<const T, L>{v.data(),          v.numChannels(),               \
                                               v.tagNameHandle(), v.tagOffsetHandle(),           \
                                               v.tagSizeHandle(), v.numProperties()};            \
  }

  INSTANTIATE_TILE_VECTOR_VIEWLITE(int, 8);
  INSTANTIATE_TILE_VECTOR_VIEWLITE(int, 32);
  INSTANTIATE_TILE_VECTOR_VIEWLITE(int, 64);
  INSTANTIATE_TILE_VECTOR_VIEWLITE(int, 512);
  INSTANTIATE_TILE_VECTOR_VIEWLITE(float, 8);
  INSTANTIATE_TILE_VECTOR_VIEWLITE(float, 32);
  INSTANTIATE_TILE_VECTOR_VIEWLITE(float, 64);
  INSTANTIATE_TILE_VECTOR_VIEWLITE(float, 512);
  INSTANTIATE_TILE_VECTOR_VIEWLITE(double, 8);
  INSTANTIATE_TILE_VECTOR_VIEWLITE(double, 32);
  INSTANTIATE_TILE_VECTOR_VIEWLITE(double, 64);
  INSTANTIATE_TILE_VECTOR_VIEWLITE(double, 512);

}  // namespace zs
