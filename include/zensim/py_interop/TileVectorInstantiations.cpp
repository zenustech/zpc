#include "zensim/container/TileVector.hpp"
#include "zensim/py_interop/TileVectorView.hpp"

extern "C" {

#define INSTANTIATE_TILE_VECTOR_VIEWLITE(T, L)                                                    \
  zs::TileVectorViewLite<T, L> pyview##T##L(zs::TileVector<T, L, zs::ZSPmrAllocator<false>> &v) { \
    return zs::TileVectorViewLite<T, L>{v.data(), v.numChannels()};                               \
  }                                                                                               \
  zs::TileVectorViewLite<const T, L> pyview##T##L##const(                                         \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<false>> &v) {                                 \
    return zs::TileVectorViewLite<const T, L>{v.data(), v.numChannels()};                         \
  }                                                                                               \
  zs::TileVectorViewLite<T, L> pyview##T##L##virtual(                                             \
      zs::TileVector<T, L, zs::ZSPmrAllocator<true>> & v) {                                       \
    return zs::TileVectorViewLite<T, L>{v.data(), v.numChannels()};                               \
  }                                                                                               \
  zs::TileVectorViewLite<const T, L> pyview##T##L##const##virtual(                                \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<true>> &v) {                                  \
    return zs::TileVectorViewLite<const T, L>{v.data(), v.numChannels()};                         \
  }                                                                                               \
  zs::TileVectorNamedViewLite<T, L> pyview##T##L##name(                                           \
      zs::true_type, zs::TileVector<T, L, zs::ZSPmrAllocator<false>> &v) {                            \
    return zs::TileVectorNamedViewLite<T, L>{v.data(),          v.numChannels(),                  \
                                             v.tagNameHandle(), v.tagOffsetHandle(),              \
                                             v.tagSizeHandle(), v.numProperties()};               \
  }                                                                                               \
  zs::TileVectorNamedViewLite<const T, L> pyview##T##L##name##const(                              \
      zs::true_type, const zs::TileVector<T, L, zs::ZSPmrAllocator<false>> &v) {                      \
    return zs::TileVectorNamedViewLite<const T, L>{v.data(),          v.numChannels(),            \
                                                   v.tagNameHandle(), v.tagOffsetHandle(),        \
                                                   v.tagSizeHandle(), v.numProperties()};         \
  }                                                                                               \
  zs::TileVectorNamedViewLite<T, L> pyview##T##L##name##virtual(                                  \
      zs::true_type, zs::TileVector<T, L, zs::ZSPmrAllocator<true>> & v) {                            \
    return zs::TileVectorNamedViewLite<T, L>{v.data(),          v.numChannels(),                  \
                                             v.tagNameHandle(), v.tagOffsetHandle(),              \
                                             v.tagSizeHandle(), v.numProperties()};               \
  }                                                                                               \
  zs::TileVectorNamedViewLite<const T, L> pyview##T##L##name##const##virtual(                     \
      zs::true_type, const zs::TileVector<T, L, zs::ZSPmrAllocator<true>> &v) {                       \
    return zs::TileVectorNamedViewLite<const T, L>{v.data(),          v.numChannels(),            \
                                                   v.tagNameHandle(), v.tagOffsetHandle(),        \
                                                   v.tagSizeHandle(), v.numProperties()};         \
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
