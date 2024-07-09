#include "zensim/container/TileVector.hpp"
#include "zensim/py_interop/GenericIterator.hpp"
#include "zensim/py_interop/TileVectorView.hpp"

extern "C" {

/* tags */
ZPC_EXPORT std::vector<zs::PropertyTag> *property_tags(const char *names[], int sizes[],
                                                       zs::size_t numTags) {
  auto ret = new std::vector<zs::PropertyTag>(numTags);
  auto &tmp = *ret;
  for (zs::size_t i = 0; i != numTags; ++i)
    tmp[i] = zs::PropertyTag{zs::SmallString{names[i]}, sizes[i]};
  return ret;
}
ZPC_EXPORT void del_property_tags(std::vector<zs::PropertyTag> *v) { delete v; }
ZPC_EXPORT void property_tags_get_item(std::vector<zs::PropertyTag> *v, zs::size_t index,
                                       const char **name, zs::size_t *size) {
  *name = (*v)[index].name;
  *size = (*v)[index].numChannels;
}
ZPC_EXPORT zs::size_t property_tags_get_size(std::vector<zs::PropertyTag> *v) { return v->size(); }

#define INSTANTIATE_TILE_VECTOR_CAPIS(T, L)                                                        \
  /* container */                                                                                  \
  ZPC_EXPORT zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *container##__##tv##_##T##_##L(       \
      const zs::ZSPmrAllocator<false> *allocator, const std::vector<zs::PropertyTag> *tags,        \
      zs::size_t size) {                                                                           \
    return new zs::TileVector<T, L, zs::ZSPmrAllocator<false>>{*allocator, *tags, size};           \
  }                                                                                                \
  ZPC_EXPORT zs::TileVector<T, L, zs::ZSPmrAllocator<true>>                                        \
      *container##__##tv##_##T##_##L##_##virtual(const zs::ZSPmrAllocator<true> *allocator,        \
                                                 const std::vector<zs::PropertyTag> *tags,         \
                                                 zs::size_t size) {                                \
    return new zs::TileVector<T, L, zs::ZSPmrAllocator<true>>{*allocator, *tags, size};            \
  }                                                                                                \
  ZPC_EXPORT void del_container##__##tv##_##T##_##L(                                               \
      zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v) {                                        \
    delete v;                                                                                      \
  }                                                                                                \
  ZPC_EXPORT void del_container##__##tv##_##T##_##L##_##virtual(                                   \
      zs::TileVector<T, L, zs::ZSPmrAllocator<true>> * v) {                                        \
    delete v;                                                                                      \
  }                                                                                                \
  ZPC_EXPORT void relocate_container##__##tv##_##T##_##L(                                          \
      zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v, zs::memsrc_e mre, zs::ProcID devid) {    \
    *v = v->clone({mre, devid});                                                                   \
  }                                                                                                \
  ZPC_EXPORT void relocate_container##__##tv##_##T##_##L##_##virtual(                              \
      zs::TileVector<T, L, zs::ZSPmrAllocator<true>> * v, zs::memsrc_e mre, zs::ProcID devid) {    \
    *v = v->clone({mre, devid});                                                                   \
  }                                                                                                \
  ZPC_EXPORT void resize_container##__##tv##_##T##_##L(                                            \
      zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v, zs::size_t newSize) {                    \
    v->resize(newSize);                                                                            \
  }                                                                                                \
  ZPC_EXPORT void resize_container##__##tv##_##T##_##L##_##virtual(                                \
      zs::TileVector<T, L, zs::ZSPmrAllocator<true>> * v, zs::size_t newSize) {                    \
    v->resize(newSize);                                                                            \
  }                                                                                                \
  ZPC_EXPORT void reset_container##__##tv##_##T##_##L(                                             \
      zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v, int ch) {                                \
    v->reset(ch);                                                                                  \
  }                                                                                                \
  ZPC_EXPORT void reset_container##__##tv##_##T##_##L##_##virtual(                                 \
      zs::TileVector<T, L, zs::ZSPmrAllocator<true>> * v, int ch) {                                \
    v->reset(ch);                                                                                  \
  }                                                                                                \
  ZPC_EXPORT size_t container_size##__##tv##_##T##_##L(                                            \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v) {                                  \
    return v->size();                                                                              \
  }                                                                                                \
  ZPC_EXPORT size_t container_size##__##tv##_##T##_##L##_##virtual(                                \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<true>> *v) {                                   \
    return v->size();                                                                              \
  }                                                                                                \
  ZPC_EXPORT size_t container_capacity##__##tv##_##T##_##L(                                        \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v) {                                  \
    return v->capacity();                                                                          \
  }                                                                                                \
  ZPC_EXPORT size_t container_capacity##__##tv##_##T##_##L##_##virtual(                            \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<true>> *v) {                                   \
    return v->capacity();                                                                          \
  }                                                                                                \
  /* iterator */                                                                                   \
  ZPC_EXPORT aosoa_iterator_port_##T##_1 get_iterator_1##__##tv##_##T##_##L(                       \
      zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v, zs::u32 id, zs::u32 chnOffset) {         \
    return aosoa_iter_##T##_1{                                                                     \
        zs::wrapv<zs::layout_e::aosoa>{}, v->data(), id, (zs::u32)L, chnOffset,                    \
        (zs::u32)v->numChannels()};                                                                \
  }                                                                                                \
  ZPC_EXPORT aosoa_iterator_port_const_##T##_1 get_iterator_1##__##tv##_##const##_##T##_##L(       \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v, zs::u32 id, zs::u32 chnOffset) {   \
    return aosoa_iter_const_##T##_1{                                                               \
        zs::wrapv<zs::layout_e::aosoa>{}, v->data(), id, (zs::u32)L, chnOffset,                    \
        (zs::u32)v->numChannels()};                                                                \
  }                                                                                                \
  ZPC_EXPORT aosoa_iterator_port_##T##_1 get_iterator_1##__##tv##_##T##_##L##_##virtual(           \
      zs::TileVector<T, L, zs::ZSPmrAllocator<true>> * v, zs::u32 id, zs::u32 chnOffset) {         \
    return aosoa_iter_##T##_1{                                                                     \
        zs::wrapv<zs::layout_e::aosoa>{}, v->data(), id, (zs::u32)L, chnOffset,                    \
        (zs::u32)v->numChannels()};                                                                \
  }                                                                                                \
  ZPC_EXPORT aosoa_iterator_port_const_##T##_1                                                     \
      get_iterator_1##__##tv##_##const##_##T##_##L##_##virtual(                                    \
          const zs::TileVector<T, L, zs::ZSPmrAllocator<true>> *v, zs::u32 id,                     \
          zs::u32 chnOffset) {                                                                     \
    return aosoa_iter_const_##T##_1{                                                               \
        zs::wrapv<zs::layout_e::aosoa>{}, v->data(), id, (zs::u32)L, chnOffset,                    \
        (zs::u32)v->numChannels()};                                                                \
  }                                                                                                \
  ZPC_EXPORT aosoa_iterator_port_##T##_3 get_iterator_3##__##tv##_##T##_##L(                       \
      zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v, zs::u32 id, zs::u32 chnOffset) {         \
    return aosoa_iter_##T##_3{                                                                     \
        zs::wrapv<zs::layout_e::aosoa>{}, v->data(), id, (zs::u32)L, chnOffset,                    \
        (zs::u32)v->numChannels()};                                                                \
  }                                                                                                \
  ZPC_EXPORT aosoa_iterator_port_const_##T##_3 get_iterator_3##__##tv##_##const##_##T##_##L(       \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v, zs::u32 id, zs::u32 chnOffset) {   \
    return aosoa_iter_const_##T##_3{                                                               \
        zs::wrapv<zs::layout_e::aosoa>{}, v->data(), id, (zs::u32)L, chnOffset,                    \
        (zs::u32)v->numChannels()};                                                                \
  }                                                                                                \
  ZPC_EXPORT aosoa_iterator_port_##T##_3 get_iterator_3##__##tv##_##T##_##L##_##virtual(           \
      zs::TileVector<T, L, zs::ZSPmrAllocator<true>> * v, zs::u32 id, zs::u32 chnOffset) {         \
    return aosoa_iter_##T##_3{                                                                     \
        zs::wrapv<zs::layout_e::aosoa>{}, v->data(), id, (zs::u32)L, chnOffset,                    \
        (zs::u32)v->numChannels()};                                                                \
  }                                                                                                \
  ZPC_EXPORT aosoa_iterator_port_const_##T##_3                                                     \
      get_iterator_3##__##tv##_##const##_##T##_##L##_##virtual(                                    \
          const zs::TileVector<T, L, zs::ZSPmrAllocator<true>> *v, zs::u32 id,                     \
          zs::u32 chnOffset) {                                                                     \
    return aosoa_iter_const_##T##_3{                                                               \
        zs::wrapv<zs::layout_e::aosoa>{}, v->data(), id, (zs::u32)L, chnOffset,                    \
        (zs::u32)v->numChannels()};                                                                \
  }                                                                                                \
  /* custom */                                                                                     \
  ZPC_EXPORT int property_offset##__##tv##_##T##_##L(                                              \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v, const char *tag) {                 \
    return v->getPropertyOffset(tag);                                                              \
  }                                                                                                \
  ZPC_EXPORT int property_offset##__##tv##_##T##_##L##_##virtual(                                  \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<true>> *v, const char *tag) {                  \
    return v->getPropertyOffset(tag);                                                              \
  }                                                                                                \
  ZPC_EXPORT int property_size##__##tv##_##T##_##L(                                                \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v, const char *tag) {                 \
    return v->getPropertySize(tag);                                                                \
  }                                                                                                \
  ZPC_EXPORT int property_size##__##tv##_##T##_##L##_##virtual(                                    \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<true>> *v, const char *tag) {                  \
    return v->getPropertySize(tag);                                                                \
  }                                                                                                \
  /* pyview */                                                                                     \
  ZPC_EXPORT zs::TileVectorViewLite<T, L> *pyview##__##tv##_##T##_##L(                             \
      zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v) {                                        \
    return new zs::TileVectorViewLite<T, L>{v->data(), v->numChannels()};                          \
  }                                                                                                \
  ZPC_EXPORT zs::TileVectorViewLite<const T, L> *pyview##__##tv##_##const##_##T##_##L(             \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v) {                                  \
    return new zs::TileVectorViewLite<const T, L>{v->data(), v->numChannels()};                    \
  }                                                                                                \
  ZPC_EXPORT zs::TileVectorViewLite<T, L> *pyview##__##tv##_##T##_##L##_##virtual(                 \
      zs::TileVector<T, L, zs::ZSPmrAllocator<true>> * v) {                                        \
    return new zs::TileVectorViewLite<T, L>{v->data(), v->numChannels()};                          \
  }                                                                                                \
  ZPC_EXPORT zs::TileVectorViewLite<const T, L> *pyview##__##tv##_##const##_##T##_##L##_##virtual( \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<true>> *v) {                                   \
    return new zs::TileVectorViewLite<const T, L>{v->data(), v->numChannels()};                    \
  }                                                                                                \
  ZPC_EXPORT zs::TileVectorNamedViewLite<T, L> *pyview##__##tvn##_##T##_##L(                       \
      zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v) {                                        \
    return new zs::TileVectorNamedViewLite<T, L>{v->data(),          v->numChannels(),             \
                                                 v->tagNameHandle(), v->tagOffsetHandle(),         \
                                                 v->tagSizeHandle(), v->numProperties()};          \
  }                                                                                                \
  ZPC_EXPORT zs::TileVectorNamedViewLite<const T, L> *pyview##__##tvn##_##const##_##T##_##L(       \
      const zs::TileVector<T, L, zs::ZSPmrAllocator<false>> *v) {                                  \
    return new zs::TileVectorNamedViewLite<const T, L>{v->data(),          v->numChannels(),       \
                                                       v->tagNameHandle(), v->tagOffsetHandle(),   \
                                                       v->tagSizeHandle(), v->numProperties()};    \
  }                                                                                                \
  ZPC_EXPORT zs::TileVectorNamedViewLite<T, L> *pyview##__##tvn##_##T##_##L##_##virtual(           \
      zs::TileVector<T, L, zs::ZSPmrAllocator<true>> * v) {                                        \
    return new zs::TileVectorNamedViewLite<T, L>{v->data(),          v->numChannels(),             \
                                                 v->tagNameHandle(), v->tagOffsetHandle(),         \
                                                 v->tagSizeHandle(), v->numProperties()};          \
  }                                                                                                \
  ZPC_EXPORT zs::TileVectorNamedViewLite<const T, L>                                               \
      *pyview##__##tvn##_##const##_##T##_##L##_##virtual(                                          \
          const zs::TileVector<T, L, zs::ZSPmrAllocator<true>> *v) {                               \
    return new zs::TileVectorNamedViewLite<const T, L>{v->data(),          v->numChannels(),       \
                                                       v->tagNameHandle(), v->tagOffsetHandle(),   \
                                                       v->tagSizeHandle(), v->numProperties()};    \
  }                                                                                                \
  ZPC_EXPORT void del_pyview##__##tv##_##T##_##L(zs::TileVectorViewLite<T, L> *v) { delete v; }    \
  ZPC_EXPORT void del_pyview##__##tv##_##const##_##T##_##L(                                        \
      zs::TileVectorViewLite<const T, L> *v) {                                                     \
    delete v;                                                                                      \
  }                                                                                                \
  ZPC_EXPORT void del_pyview##__##tvn##_##T##_##L(zs::TileVectorNamedViewLite<T, L> *v) {          \
    delete v;                                                                                      \
  }                                                                                                \
  ZPC_EXPORT void del_pyview##__##tvn##_##const##_##T##_##L(                                       \
      zs::TileVectorNamedViewLite<const T, L> *v) {                                                \
    delete v;                                                                                      \
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