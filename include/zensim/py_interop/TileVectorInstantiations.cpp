#include "zensim/container/TileVector.hpp"
#include "zensim/py_interop/TileVectorView.hpp"

extern "C" {

/* tags */
std::vector<zs::PropertyTag> *property_tags(const char *names[], int sizes[], zs::size_t numTags) {
  auto ret = new std::vector<zs::PropertyTag>(numTags);
  for (zs::size_t i = 0; i != numTags; ++i)
    (*ret)[i] = zs::PropertyTag{zs::SmallString{names[i]}, sizes[i]};
  return ret;
}
void del_property_tags(std::vector<zs::PropertyTag> *v) { delete v; }

#define INSTANTIATE_TILE_VECTOR_CAPIS(T, L)                                                       \
  /* container */                                                                                 \
  zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *container##__##tv##_##T##_##L(                 \
      const zs::ZSPmrAllocator<false> *allocator, const std::vector<zs::PropertyTag> *tags,       \
      zs::size_t size) {                                                                          \
    return new zs::TileVector<T, L, zs::ZSPmrAllocator<false>>{*allocator, *tags, size};          \
  }                                                                                               \
  zs::TileVector<T, L, zs::ZSPmrAllocator<true>> *container##__##tv##_##T##_##L##_##virtual(      \
      const zs::ZSPmrAllocator<true> *allocator, const std::vector<zs::PropertyTag> *tags,        \
      zs::size_t size) {                                                                          \
    return new zs::TileVector<T, L, zs::ZSPmrAllocator<true>>{*allocator, *tags, size};           \
  }                                                                                               \
  void del_container##__##tv##_##T##_##L(zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v) {    \
    delete v;                                                                                     \
  }                                                                                               \
  void del_container##__##tv##_##T##_##L##_##virtual(                                             \
      zs::TileVector<T, L, zs::ZSPmrAllocator<true>> * v) {                                       \
    delete v;                                                                                     \
  }                                                                                               \
  void relocate_container##__##tv##_##T##_##L(zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v, \
                                              zs::memsrc_e mre, zs::ProcID devid) {               \
    *v = v->clone({mre, devid});                                                                  \
  }                                                                                               \
  void relocate_container##__##tv##_##T##_##L##_##virtual(                                        \
      zs::TileVector<T, L, zs::ZSPmrAllocator<true>> * v, zs::memsrc_e mre, zs::ProcID devid) {   \
    *v = v->clone({mre, devid});                                                                  \
  }                                                                                               \
  /* pyview */                                                                                    \
  zs::TileVectorViewLite<T, L> *pyview##__##tv##_##T##_##L(                                       \
      zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v) {                                       \
    return new zs::TileVectorViewLite<T, L>{v->data(), v->numChannels()};                         \
  }                                                                                               \
  zs::TileVectorViewLite<const T, L> *pyview##__##tv##_##const##_##T##_##L(                       \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v) {                                 \
    return new zs::TileVectorViewLite<const T, L>{v->data(), v->numChannels()};                   \
  }                                                                                               \
  zs::TileVectorViewLite<T, L> *pyview##__##tv##_##T##_##L##_##virtual(                           \
      zs::TileVector<T, L, zs::ZSPmrAllocator<true>> * v) {                                       \
    return new zs::TileVectorViewLite<T, L>{v->data(), v->numChannels()};                         \
  }                                                                                               \
  zs::TileVectorViewLite<const T, L> *pyview##__##tv##_##const##_##T##_##L##_##virtual(           \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<true>> *v) {                                  \
    return new zs::TileVectorViewLite<const T, L>{v->data(), v->numChannels()};                   \
  }                                                                                               \
  zs::TileVectorNamedViewLite<T, L> *pyview##__##tvn##_##T##_##L(                                 \
      zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v) {                                       \
    return new zs::TileVectorNamedViewLite<T, L>{v->data(),          v->numChannels(),            \
                                                 v->tagNameHandle(), v->tagOffsetHandle(),        \
                                                 v->tagSizeHandle(), v->numProperties()};         \
  }                                                                                               \
  zs::TileVectorNamedViewLite<const T, L> *pyview##__##tvn##_##const##_##T##_##L(                 \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v) {                                 \
    return new zs::TileVectorNamedViewLite<const T, L>{v->data(),          v->numChannels(),      \
                                                       v->tagNameHandle(), v->tagOffsetHandle(),  \
                                                       v->tagSizeHandle(), v->numProperties()};   \
  }                                                                                               \
  zs::TileVectorNamedViewLite<T, L> *pyview##__##tvn##_##T##_##L##_##virtual(                     \
      zs::TileVector<T, L, zs::ZSPmrAllocator<true>> * v) {                                       \
    return new zs::TileVectorNamedViewLite<T, L>{v->data(),          v->numChannels(),            \
                                                 v->tagNameHandle(), v->tagOffsetHandle(),        \
                                                 v->tagSizeHandle(), v->numProperties()};         \
  }                                                                                               \
  zs::TileVectorNamedViewLite<const T, L> *pyview##__##tvn##_##const##_##T##_##L##_##virtual(     \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<true>> *v) {                                  \
    return new zs::TileVectorNamedViewLite<const T, L>{v->data(),          v->numChannels(),      \
                                                       v->tagNameHandle(), v->tagOffsetHandle(),  \
                                                       v->tagSizeHandle(), v->numProperties()};   \
  }                                                                                               \
  void del_pyview##__##tv##_##T##_##L(zs::TileVectorViewLite<T, L> *v) { delete v; }              \
  void del_pyview##__##tv##_##const##_##T##_##L(zs::TileVectorViewLite<const T, L> *v) {          \
    delete v;                                                                                     \
  }                                                                                               \
  void del_pyview##__##tvn##_##T##_##L(zs::TileVectorNamedViewLite<T, L> *v) { delete v; }        \
  void del_pyview##__##tvn##_##const##_##T##_##L(zs::TileVectorNamedViewLite<const T, L> *v) {    \
    delete v;                                                                                     \
  }

INSTANTIATE_TILE_VECTOR_CAPIS(int, 8)
INSTANTIATE_TILE_VECTOR_CAPIS(int, 32)
INSTANTIATE_TILE_VECTOR_CAPIS(int, 64)
INSTANTIATE_TILE_VECTOR_CAPIS(int, 512)
INSTANTIATE_TILE_VECTOR_CAPIS(float, 8)
INSTANTIATE_TILE_VECTOR_CAPIS(float, 32)
INSTANTIATE_TILE_VECTOR_CAPIS(float, 64)
INSTANTIATE_TILE_VECTOR_CAPIS(float, 512)
INSTANTIATE_TILE_VECTOR_CAPIS(double, 8)
INSTANTIATE_TILE_VECTOR_CAPIS(double, 32)
INSTANTIATE_TILE_VECTOR_CAPIS(double, 64)
INSTANTIATE_TILE_VECTOR_CAPIS(double, 512)
}