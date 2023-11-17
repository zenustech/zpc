#include "zensim/container/DenseField.hpp"
#include "zensim/math/Vec.h"
#include "zensim/py_interop/DenseFieldView.hpp"
#include "zensim/py_interop/GenericIterator.hpp"

extern "C" {

std::vector<zs::size_t> *shape_dims(int sizes[], zs::size_t dims) {
  auto ret = new std::vector<zs::size_t>(dims);
  auto &tmp = *ret;
  for (zs::size_t i = 0; i != dims; ++i) tmp[i] = sizes[i];
  return ret;
}
void del_shape_dims(std::vector<zs::size_t> *v) { delete v; }
zs::size_t shape_dims_get_item(std::vector<zs::size_t> *v, zs::size_t index) { return (*v)[index]; }
zs::size_t shape_dims_get_size(std::vector<zs::size_t> *v) { return v->size(); }

#define INSTANTIATE_DENSE_FIELD_CAPIS(T)                                                           \
  /* container */                                                                                  \
  zs::DenseField<T, zs::ZSPmrAllocator<false>> *container##__##df##_##T(                           \
      const zs::ZSPmrAllocator<false> *allocator, const std::vector<zs::size_t> *shape) {          \
    return new zs::DenseField<T, zs::ZSPmrAllocator<false>>{*allocator, *shape};                   \
  }                                                                                                \
  zs::DenseField<T, zs::ZSPmrAllocator<true>> *container##__##df##_##T##_##virtual(                \
      const zs::ZSPmrAllocator<true> *allocator, const std::vector<zs::size_t> *shape) {           \
    return new zs::DenseField<T, zs::ZSPmrAllocator<true>>{*allocator, *shape};                    \
  }                                                                                                \
  void del_container##__##df##_##T(zs::DenseField<T, zs::ZSPmrAllocator<false>> *v) { delete v; }  \
  void del_container##__##df##_##T##_##virtual(zs::DenseField<T, zs::ZSPmrAllocator<true>> * v) {  \
    delete v;                                                                                      \
  }                                                                                                \
  void relocate_container##__##df##_##T(zs::DenseField<T, zs::ZSPmrAllocator<false>> *v,           \
                                        zs::memsrc_e mre, zs::ProcID devid) {                      \
    *v = v->clone({mre, devid});                                                                   \
  }                                                                                                \
  void relocate_container##__##df##_##T##_##virtual(                                               \
      zs::DenseField<T, zs::ZSPmrAllocator<true>> * v, zs::memsrc_e mre, zs::ProcID devid) {       \
    *v = v->clone({mre, devid});                                                                   \
  }                                                                                                \
  void resize_container##__##df##_##T(zs::DenseField<T, zs::ZSPmrAllocator<false>> *v,             \
                                      const std::vector<zs::size_t> *shape) {                      \
    v->reshape(*shape);                                                                            \
  }                                                                                                \
  void resize_container##__##df##_##T##_##virtual(zs::DenseField<T, zs::ZSPmrAllocator<true>> * v, \
                                                  const std::vector<zs::size_t> *shape) {          \
    v->reshape(*shape);                                                                            \
  }                                                                                                \
  void reset_container##__##df##_##T(zs::DenseField<T, zs::ZSPmrAllocator<false>> *v, int ch) {    \
    v->reset(ch);                                                                                  \
  }                                                                                                \
  void reset_container##__##df##_##T##_##virtual(zs::DenseField<T, zs::ZSPmrAllocator<true>> * v,  \
                                                 int ch) {                                         \
    v->reset(ch);                                                                                  \
  }                                                                                                \
  size_t container_size##__##df##_##T(const zs::DenseField<T, zs::ZSPmrAllocator<false>> *v) {     \
    return v->size();                                                                              \
  }                                                                                                \
  size_t container_size##__##df##_##T##_##virtual(                                                 \
      const zs::DenseField<T, zs::ZSPmrAllocator<true>> *v) {                                      \
    return v->size();                                                                              \
  }                                                                                                \
  size_t container_capacity##__##df##_##T(const zs::DenseField<T, zs::ZSPmrAllocator<false>> *v) { \
    return v->capacity();                                                                          \
  }                                                                                                \
  size_t container_capacity##__##df##_##T##_##virtual(                                             \
      const zs::DenseField<T, zs::ZSPmrAllocator<true>> *v) {                                      \
    return v->capacity();                                                                          \
  }                                                                                                \
  /* iterator */                                                                                   \
  aosoa_iterator_##T##_1 get_iterator_1##__##df##_##T(                                             \
      zs::DenseField<T, zs::ZSPmrAllocator<false>> *v, zs::u32 id) {                               \
    return aosoa_iterator_##T##_1{zs::wrapv<zs::layout_e::aos>{}, v->data(), id};                  \
  }                                                                                                \
  aosoa_iterator_const_##T##_1 get_iterator_1##__##df##_##const##_##T(                             \
      const zs::DenseField<T, zs::ZSPmrAllocator<false>> *v, zs::u32 id) {                         \
    return aosoa_iterator_const_##T##_1{zs::wrapv<zs::layout_e::aos>{}, v->data(), id};            \
  }                                                                                                \
  aosoa_iterator_##T##_1 get_iterator_1##__##df##_##T##_##virtual(                                 \
      zs::DenseField<T, zs::ZSPmrAllocator<true>> * v, zs::u32 id) {                               \
    return aosoa_iterator_##T##_1{zs::wrapv<zs::layout_e::aos>{}, v->data(), id};                  \
  }                                                                                                \
  aosoa_iterator_const_##T##_1 get_iterator_1##__##df##_##const##_##T##_##virtual(                 \
      const zs::DenseField<T, zs::ZSPmrAllocator<true>> *v, zs::u32 id) {                          \
    return aosoa_iterator_const_##T##_1{zs::wrapv<zs::layout_e::aos>{}, v->data(), id};            \
  }                                                                                                \
  aosoa_iterator_##T##_3 get_iterator_3##__##df##_##T(                                             \
      zs::DenseField<T, zs::ZSPmrAllocator<false>> *v, zs::u32 id) {                               \
    return aosoa_iterator_##T##_3{zs::wrapv<zs::layout_e::aos>{}, v->data(), id};                  \
  }                                                                                                \
  aosoa_iterator_const_##T##_3 get_iterator_3##__##df##_##const##_##T(                             \
      const zs::DenseField<T, zs::ZSPmrAllocator<false>> *v, zs::u32 id) {                         \
    return aosoa_iterator_const_##T##_3{zs::wrapv<zs::layout_e::aos>{}, v->data(), id};            \
  }                                                                                                \
  aosoa_iterator_##T##_3 get_iterator_3##__##df##_##T##_##virtual(                                 \
      zs::DenseField<T, zs::ZSPmrAllocator<true>> * v, zs::u32 id) {                               \
    return aosoa_iterator_##T##_3{zs::wrapv<zs::layout_e::aos>{}, v->data(), id};                  \
  }                                                                                                \
  aosoa_iterator_const_##T##_3 get_iterator_3##__##df##_##const##_##T##_##virtual(                 \
      const zs::DenseField<T, zs::ZSPmrAllocator<true>> *v, zs::u32 id) {                          \
    return aosoa_iterator_const_##T##_3{zs::wrapv<zs::layout_e::aos>{}, v->data(), id};            \
  }                                                                                                \
  /* custom */                                                                                     \
  T get_val_container##__##df##_##T(zs::DenseField<T, zs::ZSPmrAllocator<false>> *v) {             \
    return v->getVal();                                                                            \
  }                                                                                                \
  void set_val_container##__##df##_##T(zs::DenseField<T, zs::ZSPmrAllocator<false>> *v,            \
                                       T newVal) {                                                 \
    v->setVal(newVal);                                                                             \
  }                                                                                                \
  /* pyview */                                                                                     \
  zs::DenseFieldViewLite<T> *pyview##__##df##_##T(                                                 \
      zs::DenseField<T, zs::ZSPmrAllocator<false>> *v) {                                           \
    return new zs::DenseFieldViewLite<T>{v->data(), v->_shape.data(), v->dims()};                  \
  }                                                                                                \
  zs::DenseFieldViewLite<const T> *pyview##__##df##_##const##_##T(                                 \
      const zs::DenseField<T, zs::ZSPmrAllocator<false>> *v) {                                     \
    return new zs::DenseFieldViewLite<const T>{v->data(), v->_shape.data(), v->dims()};            \
  }                                                                                                \
  zs::DenseFieldViewLite<T> *pyview##__##df##_##T##_##virtual(                                     \
      zs::DenseField<T, zs::ZSPmrAllocator<true>> * v) {                                           \
    return new zs::DenseFieldViewLite<T>{v->data(), v->_shape.data(), v->dims()};                  \
  }                                                                                                \
  zs::DenseFieldViewLite<const T> *pyview##__##df##_##const##_##T##_##virtual(                     \
      const zs::DenseField<T, zs::ZSPmrAllocator<true>> *v) {                                      \
    return new zs::DenseFieldViewLite<const T>{v->data(), v->_shape.data(), v->dims()};            \
  }                                                                                                \
  void del_pyview##__##df##_##T(zs::DenseFieldViewLite<T> *v) { delete v; }                        \
  void del_pyview##__##df##_##const##_##T(zs::DenseFieldViewLite<const T> *v) { delete v; }

INSTANTIATE_DENSE_FIELD_CAPIS(int)
INSTANTIATE_DENSE_FIELD_CAPIS(float)
INSTANTIATE_DENSE_FIELD_CAPIS(double)

#if 0
INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<int, 2>)
INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<int, 3>)
INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<int, 4>)
INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<float, 2>)
INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<float, 3>)
INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<float, 4>)
INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<double, 2>)
INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<double, 3>)
INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<double, 4>)

INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<int, 2, 2>)
INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<int, 3, 3>)
INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<int, 4, 4>)
INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<float, 2, 2>)
INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<float, 3, 3>)
INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<float, 4, 4>)
INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<double, 2, 2>)
INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<double, 3, 3>)
INSTANTIATE_DENSE_FIELD_CAPIS(zs::vec<double, 4, 4>)

#endif
}  // namespace zs
