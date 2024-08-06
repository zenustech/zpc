#include "zensim/container/Vector.hpp"
#include "zensim/math/Vec.h"
#include "zensim/py_interop/GenericIterator.hpp"
#include "zensim/py_interop/VectorView.hpp"

extern "C" {

#define INSTANTIATE_VECTOR_CAPIS(T)                                                               \
  /* container */                                                                                 \
  ZPC_EXPORT zs::Vector<T, zs::ZSPmrAllocator<false>> *container##__##v##_##T(                    \
      const zs::ZSPmrAllocator<false> *allocator, zs::size_t size) {                              \
    return new zs::Vector<T, zs::ZSPmrAllocator<false>>{*allocator, size};                        \
  }                                                                                               \
  ZPC_EXPORT zs::Vector<T, zs::ZSPmrAllocator<true>> *container##__##v##_##T##_##virtual(         \
      const zs::ZSPmrAllocator<true> *allocator, zs::size_t size) {                               \
    return new zs::Vector<T, zs::ZSPmrAllocator<true>>{*allocator, size};                         \
  }                                                                                               \
  ZPC_EXPORT void del_container##__##v##_##T(zs::Vector<T, zs::ZSPmrAllocator<false>> *v) {       \
    delete v;                                                                                     \
  }                                                                                               \
  ZPC_EXPORT void del_container##__##v##_##T##_##virtual(zs::Vector<T, zs::ZSPmrAllocator<true>>  \
                                                         * v) {                                   \
    delete v;                                                                                     \
  }                                                                                               \
  ZPC_EXPORT void relocate_container##__##v##_##T(zs::Vector<T, zs::ZSPmrAllocator<false>> *v,    \
                                                  zs::memsrc_e mre, zs::ProcID devid) {           \
    *v = v->clone({mre, devid});                                                                  \
  }                                                                                               \
  ZPC_EXPORT void relocate_container##__##v##_##T##_##virtual(                                    \
      zs::Vector<T, zs::ZSPmrAllocator<true>> * v, zs::memsrc_e mre, zs::ProcID devid) {          \
    *v = v->clone({mre, devid});                                                                  \
  }                                                                                               \
  ZPC_EXPORT void resize_container##__##v##_##T(zs::Vector<T, zs::ZSPmrAllocator<false>> *v,      \
                                                zs::size_t newSize) {                             \
    v->resize(newSize);                                                                           \
  }                                                                                               \
  ZPC_EXPORT void resize_container##__##v##_##T##_##virtual(                                      \
      zs::Vector<T, zs::ZSPmrAllocator<true>> * v, zs::size_t newSize) {                          \
    v->resize(newSize);                                                                           \
  }                                                                                               \
  ZPC_EXPORT void reset_container##__##v##_##T(zs::Vector<T, zs::ZSPmrAllocator<false>> *v,       \
                                               int ch) {                                          \
    v->reset(ch);                                                                                 \
  }                                                                                               \
  ZPC_EXPORT void reset_container##__##v##_##T##_##virtual(                                       \
      zs::Vector<T, zs::ZSPmrAllocator<true>> * v, int ch) {                                      \
    v->reset(ch);                                                                                 \
  }                                                                                               \
  ZPC_EXPORT size_t container_size##__##v##_##T(                                                  \
      const zs::Vector<T, zs::ZSPmrAllocator<false>> *v) {                                        \
    return v->size();                                                                             \
  }                                                                                               \
  ZPC_EXPORT size_t container_size##__##v##_##T##_##virtual(                                      \
      const zs::Vector<T, zs::ZSPmrAllocator<true>> *v) {                                         \
    return v->size();                                                                             \
  }                                                                                               \
  ZPC_EXPORT size_t container_capacity##__##v##_##T(                                              \
      const zs::Vector<T, zs::ZSPmrAllocator<false>> *v) {                                        \
    return v->capacity();                                                                         \
  }                                                                                               \
  ZPC_EXPORT size_t container_capacity##__##v##_##T##_##virtual(                                  \
      const zs::Vector<T, zs::ZSPmrAllocator<true>> *v) {                                         \
    return v->capacity();                                                                         \
  }                                                                                               \
  /* custom */                                                                                    \
  ZPC_EXPORT T get_val_container##__##v##_##T(zs::Vector<T, zs::ZSPmrAllocator<false>> *v) {      \
    return v->getVal();                                                                           \
  }                                                                                               \
  ZPC_EXPORT T get_val_container##__##v##_##T##_##virtual(zs::Vector<T, zs::ZSPmrAllocator<true>> \
                                                          * v) {                                  \
    return v->getVal();                                                                           \
  }                                                                                               \
  ZPC_EXPORT void set_val_container##__##v##_##T(zs::Vector<T, zs::ZSPmrAllocator<false>> *v,     \
                                                 T newVal) {                                      \
    v->setVal(newVal);                                                                            \
  }                                                                                               \
  ZPC_EXPORT void set_val_container##__##v##_##T##_##virtual(                                     \
      zs::Vector<T, zs::ZSPmrAllocator<true>> * v, T newVal) {                                    \
    v->setVal(newVal);                                                                            \
  }                                                                                               \
  ZPC_EXPORT T get_val_i_container##__##v##_##T(zs::Vector<T, zs::ZSPmrAllocator<false>> *v,      \
                                                zs::size_t i) {                                   \
    return v->getVal(i);                                                                          \
  }                                                                                               \
  ZPC_EXPORT T get_val_i_container##__##v##_##T##_##virtual(                                      \
      zs::Vector<T, zs::ZSPmrAllocator<true>> * v, zs::size_t i) {                                \
    return v->getVal(i);                                                                          \
  }                                                                                               \
  ZPC_EXPORT void set_val_i_container##__##v##_##T(zs::Vector<T, zs::ZSPmrAllocator<false>> *v,   \
                                                   zs::size_t i, T newVal) {                      \
    v->setVal(newVal, i);                                                                         \
  }                                                                                               \
  ZPC_EXPORT void set_val_i_container##__##v##_##T##_##virtual(                                   \
      zs::Vector<T, zs::ZSPmrAllocator<true>> * v, zs::size_t i, T newVal) {                      \
    v->setVal(newVal, i);                                                                         \
  }                                                                                               \
  ZPC_EXPORT void copy_to_container##__##v##_##T(zs::Vector<T, zs::ZSPmrAllocator<false>> *v,     \
                                                 void *src) {                                     \
    v->assignVals(static_cast<T *>(src));                                                         \
  }                                                                                               \
  ZPC_EXPORT void copy_to_container##__##v##_##T##_##virtual(                                     \
      zs::Vector<T, zs::ZSPmrAllocator<true>> * v, void *src) {                                   \
    v->assignVals(static_cast<T *>(src));                                                         \
  }                                                                                               \
  ZPC_EXPORT void copy_from_container##__##v##_##T(zs::Vector<T, zs::ZSPmrAllocator<false>> *v,   \
                                                   void *dst) {                                   \
    v->retrieveVals(static_cast<T *>(dst));                                                       \
  }                                                                                               \
  ZPC_EXPORT void copy_from_container##__##v##_##T##_##virtual(                                   \
      zs::Vector<T, zs::ZSPmrAllocator<true>> * v, void *dst) {                                   \
    v->retrieveVals(static_cast<T *>(dst));                                                       \
  }                                                                                               \
  ZPC_EXPORT T *get_handle_container##__##v##_##T(zs::Vector<T, zs::ZSPmrAllocator<false>> *v) {  \
    return v->data();                                                                             \
  }                                                                                               \
  ZPC_EXPORT T *get_handle_container##__##v##_##T##_##virtual(                                    \
      zs::Vector<T, zs::ZSPmrAllocator<true>> * v) {                                              \
    return v->data();                                                                             \
  }                                                                                               \
  /* pyview */                                                                                    \
  ZPC_EXPORT zs::VectorViewLite<T> *pyview##__##v##_##T(                                          \
      zs::Vector<T, zs::ZSPmrAllocator<false>> *v) {                                              \
    return new zs::VectorViewLite<T>{v->data()};                                                  \
  }                                                                                               \
  ZPC_EXPORT zs::VectorViewLite<const T> *pyview##__##v##_##const##_##T(                          \
      const zs::Vector<T, zs::ZSPmrAllocator<false>> *v) {                                        \
    return new zs::VectorViewLite<const T>{v->data()};                                            \
  }                                                                                               \
  ZPC_EXPORT zs::VectorViewLite<T> *pyview##__##v##_##T##_##virtual(                              \
      zs::Vector<T, zs::ZSPmrAllocator<true>> * v) {                                              \
    return new zs::VectorViewLite<T>{v->data()};                                                  \
  }                                                                                               \
  ZPC_EXPORT zs::VectorViewLite<const T> *pyview##__##v##_##const##_##T##_##virtual(              \
      const zs::Vector<T, zs::ZSPmrAllocator<true>> *v) {                                         \
    return new zs::VectorViewLite<const T>{v->data()};                                            \
  }                                                                                               \
  ZPC_EXPORT void del_pyview##__##v##_##T(zs::VectorViewLite<T> *v) { delete v; }                 \
  ZPC_EXPORT void del_pyview##__##v##_##const##_##T(zs::VectorViewLite<const T> *v) { delete v; }

#define INSTANTIATE_VECTOR_ITERATOR_CAPIS(T)                                                      \
  /* iterator */                                                                                  \
  ZPC_EXPORT aosoa_iterator_port_##T##_1 get_iterator_1##__##v##_##T(                             \
      zs::Vector<T, zs::ZSPmrAllocator<false>> *v, zs::u32 id) {                                  \
    return aosoa_iter_##T##_1{zs::wrapv<zs::layout_e::aos>{}, v->data(), id};                     \
  }                                                                                               \
  ZPC_EXPORT aosoa_iterator_port_const_##T##_1 get_iterator_1##__##v##_##const##_##T(             \
      const zs::Vector<T, zs::ZSPmrAllocator<false>> *v, zs::u32 id) {                            \
    return aosoa_iter_const_##T##_1{zs::wrapv<zs::layout_e::aos>{}, v->data(), id};               \
  }                                                                                               \
  ZPC_EXPORT aosoa_iterator_port_##T##_1 get_iterator_1##__##v##_##T##_##virtual(                 \
      zs::Vector<T, zs::ZSPmrAllocator<true>> * v, zs::u32 id) {                                  \
    return aosoa_iter_##T##_1{zs::wrapv<zs::layout_e::aos>{}, v->data(), id};                     \
  }                                                                                               \
  ZPC_EXPORT aosoa_iterator_port_const_##T##_1 get_iterator_1##__##v##_##const##_##T##_##virtual( \
      const zs::Vector<T, zs::ZSPmrAllocator<true>> *v, zs::u32 id) {                             \
    return aosoa_iter_const_##T##_1{zs::wrapv<zs::layout_e::aos>{}, v->data(), id};               \
  }                                                                                               \
  ZPC_EXPORT aosoa_iterator_port_##T##_3 get_iterator_3##__##v##_##T(                             \
      zs::Vector<T, zs::ZSPmrAllocator<false>> *v, zs::u32 id) {                                  \
    return aosoa_iter_##T##_3{zs::wrapv<zs::layout_e::aos>{}, v->data(), id};                     \
  }                                                                                               \
  ZPC_EXPORT aosoa_iterator_port_const_##T##_3 get_iterator_3##__##v##_##const##_##T(             \
      const zs::Vector<T, zs::ZSPmrAllocator<false>> *v, zs::u32 id) {                            \
    return aosoa_iter_const_##T##_3{zs::wrapv<zs::layout_e::aos>{}, v->data(), id};               \
  }                                                                                               \
  ZPC_EXPORT aosoa_iterator_port_##T##_3 get_iterator_3##__##v##_##T##_##virtual(                 \
      zs::Vector<T, zs::ZSPmrAllocator<true>> * v, zs::u32 id) {                                  \
    return aosoa_iter_##T##_3{zs::wrapv<zs::layout_e::aos>{}, v->data(), id};                     \
  }                                                                                               \
  ZPC_EXPORT aosoa_iterator_port_const_##T##_3 get_iterator_3##__##v##_##const##_##T##_##virtual( \
      const zs::Vector<T, zs::ZSPmrAllocator<true>> *v, zs::u32 id) {                             \
    return aosoa_iter_const_##T##_3{zs::wrapv<zs::layout_e::aos>{}, v->data(), id};               \
  }

INSTANTIATE_VECTOR_CAPIS(int)
INSTANTIATE_VECTOR_CAPIS(float)
INSTANTIATE_VECTOR_CAPIS(double)

INSTANTIATE_VECTOR_ITERATOR_CAPIS(int)
INSTANTIATE_VECTOR_ITERATOR_CAPIS(float)
INSTANTIATE_VECTOR_ITERATOR_CAPIS(double)

using vec2i = zs::vec<int, 2>;
using vec3i = zs::vec<int, 3>;
using vec4i = zs::vec<int, 4>;
using vec2f = zs::vec<float, 2>;
using vec3f = zs::vec<float, 3>;
using vec4f = zs::vec<float, 4>;
using vec2d = zs::vec<double, 2>;
using vec3d = zs::vec<double, 3>;
using vec4d = zs::vec<double, 4>;
INSTANTIATE_VECTOR_CAPIS(vec2i)
INSTANTIATE_VECTOR_CAPIS(vec3i)
INSTANTIATE_VECTOR_CAPIS(vec4i)
INSTANTIATE_VECTOR_CAPIS(vec2f)
INSTANTIATE_VECTOR_CAPIS(vec3f)
INSTANTIATE_VECTOR_CAPIS(vec4f)
INSTANTIATE_VECTOR_CAPIS(vec2d)
INSTANTIATE_VECTOR_CAPIS(vec3d)
INSTANTIATE_VECTOR_CAPIS(vec4d)

using mat2i = zs::vec<int, 2, 2>;
using mat3i = zs::vec<int, 3, 3>;
using mat4i = zs::vec<int, 4, 4>;
using mat2f = zs::vec<float, 2, 2>;
using mat3f = zs::vec<float, 3, 3>;
using mat4f = zs::vec<float, 4, 4>;
using mat2d = zs::vec<double, 2, 2>;
using mat3d = zs::vec<double, 3, 3>;
using mat4d = zs::vec<double, 4, 4>;
INSTANTIATE_VECTOR_CAPIS(mat2i)
INSTANTIATE_VECTOR_CAPIS(mat3i)
INSTANTIATE_VECTOR_CAPIS(mat4i)
INSTANTIATE_VECTOR_CAPIS(mat2f)
INSTANTIATE_VECTOR_CAPIS(mat3f)
INSTANTIATE_VECTOR_CAPIS(mat4f)
INSTANTIATE_VECTOR_CAPIS(mat2d)
INSTANTIATE_VECTOR_CAPIS(mat3d)
INSTANTIATE_VECTOR_CAPIS(mat4d)

#if 0
INSTANTIATE_VECTOR_CAPIS(zs::vec<int, 2>)
INSTANTIATE_VECTOR_CAPIS(zs::vec<int, 3>)
INSTANTIATE_VECTOR_CAPIS(zs::vec<int, 4>)
INSTANTIATE_VECTOR_CAPIS(zs::vec<float, 2>)
INSTANTIATE_VECTOR_CAPIS(zs::vec<float, 3>)
INSTANTIATE_VECTOR_CAPIS(zs::vec<float, 4>)
INSTANTIATE_VECTOR_CAPIS(zs::vec<double, 2>)
INSTANTIATE_VECTOR_CAPIS(zs::vec<double, 3>)
INSTANTIATE_VECTOR_CAPIS(zs::vec<double, 4>)

INSTANTIATE_VECTOR_CAPIS(zs::vec<int, 2, 2>)
INSTANTIATE_VECTOR_CAPIS(zs::vec<int, 3, 3>)
INSTANTIATE_VECTOR_CAPIS(zs::vec<int, 4, 4>)
INSTANTIATE_VECTOR_CAPIS(zs::vec<float, 2, 2>)
INSTANTIATE_VECTOR_CAPIS(zs::vec<float, 3, 3>)
INSTANTIATE_VECTOR_CAPIS(zs::vec<float, 4, 4>)
INSTANTIATE_VECTOR_CAPIS(zs::vec<double, 2, 2>)
INSTANTIATE_VECTOR_CAPIS(zs::vec<double, 3, 3>)
INSTANTIATE_VECTOR_CAPIS(zs::vec<double, 4, 4>)

#endif
}  // namespace zs
