#include "zensim/container/Vector.hpp"
#include "zensim/py_interop/VectorView.hpp"

extern "C" {

#define INSTANTIATE_VECTOR_CAPIS(T)                                                              \
  /* container */                                                                                \
  zs::Vector<T, zs::ZSPmrAllocator<false>> *container##__##v##_##T(                              \
      const zs::ZSPmrAllocator<false> *allocator, zs::size_t size) {                             \
    return new zs::Vector<T, zs::ZSPmrAllocator<false>>{*allocator, size};                       \
  }                                                                                              \
  zs::Vector<T, zs::ZSPmrAllocator<true>> *container##__##v##_##T##_##virtual(                   \
      const zs::ZSPmrAllocator<true> *allocator, zs::size_t size) {                              \
    return new zs::Vector<T, zs::ZSPmrAllocator<true>>{*allocator, size};                        \
  }                                                                                              \
  void del_container##__##v##_##T(zs::Vector<T, zs::ZSPmrAllocator<false>> *v) { delete v; }     \
  void del_container##__##v##_##T##_##virtual(zs::Vector<T, zs::ZSPmrAllocator<true>> * v) {     \
    delete v;                                                                                    \
  }                                                                                              \
  void relocate_container##__##v##_##T(zs::Vector<T, zs::ZSPmrAllocator<false>> *v,              \
                                       zs::memsrc_e mre, zs::ProcID devid) {                     \
    *v = v->clone({mre, devid});                                                                 \
  }                                                                                              \
  void relocate_container##__##v##_##T##_##virtual(zs::Vector<T, zs::ZSPmrAllocator<true>> * v,  \
                                                   zs::memsrc_e mre, zs::ProcID devid) {         \
    *v = v->clone({mre, devid});                                                                 \
  }                                                                                              \
  /* pyview */                                                                                   \
  zs::VectorViewLite<T> *pyview##__##v##_##T(zs::Vector<T, zs::ZSPmrAllocator<false>> *v) {      \
    return new zs::VectorViewLite<T>{v->data()};                                                 \
  }                                                                                              \
  zs::VectorViewLite<const T> *pyview##__##v##_##const##_##T(                                    \
      const zs::Vector<T, zs::ZSPmrAllocator<false>> *v) {                                       \
    return new zs::VectorViewLite<const T>{v->data()};                                           \
  }                                                                                              \
  zs::VectorViewLite<T> *pyview##__##v##_##T##_##virtual(zs::Vector<T, zs::ZSPmrAllocator<true>> \
                                                         * v) {                                  \
    return new zs::VectorViewLite<T>{v->data()};                                                 \
  }                                                                                              \
  zs::VectorViewLite<const T> *pyview##__##v##_##const##_##T##_##virtual(                        \
      const zs::Vector<T, zs::ZSPmrAllocator<true>> *v) {                                        \
    return new zs::VectorViewLite<const T>{v->data()};                                           \
  }                                                                                              \
  void del_pyview##__##v##_##T(zs::VectorViewLite<T> *v) { delete v; }                           \
  void del_pyview##__##v##_##const##_##T(zs::VectorViewLite<const T> *v) { delete v; }

INSTANTIATE_VECTOR_CAPIS(int)
INSTANTIATE_VECTOR_CAPIS(float)
INSTANTIATE_VECTOR_CAPIS(double)

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

}  // namespace zs
