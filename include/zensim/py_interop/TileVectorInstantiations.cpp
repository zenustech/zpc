#include "zensim/container/TileVector.hpp"
#include "zensim/py_interop/TileVectorView.hpp"

extern "C" {

#define INSTANTIATE_TILE_VECTOR_VIEWLITE(T, L)                                                   \
  zs::TileVectorViewLite<T, L> *pyview##__##tv##_##T##_##L(                                      \
      zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v) {                                      \
    return new zs::TileVectorViewLite<T, L>{v->data(), v->numChannels()};                        \
  }                                                                                              \
  zs::TileVectorViewLite<const T, L> *pyview##__##tv##_##const##_##T##_##L(                      \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v) {                                \
    return new zs::TileVectorViewLite<const T, L>{v->data(), v->numChannels()};                  \
  }                                                                                              \
  zs::TileVectorViewLite<T, L> *pyview##__##tv##_##T##_##L##_##virtual(                          \
      zs::TileVector<T, L, zs::ZSPmrAllocator<true>> * v) {                                      \
    return new zs::TileVectorViewLite<T, L>{v->data(), v->numChannels()};                        \
  }                                                                                              \
  zs::TileVectorViewLite<const T, L> *pyview##__##tv##_##const##_##T##_##L##_##virtual(          \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<true>> *v) {                                 \
    return new zs::TileVectorViewLite<const T, L>{v->data(), v->numChannels()};                  \
  }                                                                                              \
  zs::TileVectorNamedViewLite<T, L> *pyview##__##tvn##_##T##_##L(                                \
      zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v) {                                      \
    return new zs::TileVectorNamedViewLite<T, L>{v->data(),          v->numChannels(),           \
                                                 v->tagNameHandle(), v->tagOffsetHandle(),       \
                                                 v->tagSizeHandle(), v->numProperties()};        \
  }                                                                                              \
  zs::TileVectorNamedViewLite<const T, L> *pyview##__##tvn##_##const##_##T##_##L(                \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v) {                                \
    return new zs::TileVectorNamedViewLite<const T, L>{v->data(),          v->numChannels(),     \
                                                       v->tagNameHandle(), v->tagOffsetHandle(), \
                                                       v->tagSizeHandle(), v->numProperties()};  \
  }                                                                                              \
  zs::TileVectorNamedViewLite<T, L> *pyview##__##tvn##_##T##_##L##_##virtual(                    \
      zs::TileVector<T, L, zs::ZSPmrAllocator<true>> * v) {                                      \
    return new zs::TileVectorNamedViewLite<T, L>{v->data(),          v->numChannels(),           \
                                                 v->tagNameHandle(), v->tagOffsetHandle(),       \
                                                 v->tagSizeHandle(), v->numProperties()};        \
  }                                                                                              \
  zs::TileVectorNamedViewLite<const T, L> *pyview##__##tvn##_##const##_##T##_##L##_##virtual(    \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<true>> *v) {                                 \
    return new zs::TileVectorNamedViewLite<const T, L>{v->data(),          v->numChannels(),     \
                                                       v->tagNameHandle(), v->tagOffsetHandle(), \
                                                       v->tagSizeHandle(), v->numProperties()};  \
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
}
