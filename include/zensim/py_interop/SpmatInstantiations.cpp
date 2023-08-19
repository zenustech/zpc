#include "zensim/math/Vec.h"
#include "zensim/math/matrix/SparseMatrix.hpp"
#include "zensim/py_interop/SmallVec.hpp"
#include "zensim/py_interop/SpmatView.hpp"

extern "C" {

#define INSTANTIATE_SPMAT_CAPIS(T, RowMajor, Ti, Tn)                                             \
  /* container */                                                                                \
  zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>>                               \
      *container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn(                                      \
          const zs::ZSPmrAllocator<false> *allocator, Ti ni, Ti nj) {                            \
    return new zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>>{*allocator, ni,  \
                                                                                nj};             \
  }                                                                                              \
  zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>>                                \
      *container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn##_##virtual(                          \
          const zs::ZSPmrAllocator<true> *allocator, Ti ni, Ti nj) {                             \
    return new zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>>{*allocator, ni,   \
                                                                               nj};              \
  }                                                                                              \
  void del_container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn(                                  \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat) {                 \
    delete spmat;                                                                                \
  }                                                                                              \
  void del_container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn##_##virtual(                      \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>> * spmat) {                 \
    delete spmat;                                                                                \
  }                                                                                              \
  void relocate_container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn(                             \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat, zs::memsrc_e mre, \
      zs::ProcID devid) {                                                                        \
    *spmat = spmat->clone({mre, devid});                                                         \
  }                                                                                              \
  void relocate_container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn##_##virtual(                 \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>> * spmat, zs::memsrc_e mre, \
      zs::ProcID devid) {                                                                        \
    *spmat = spmat->clone({mre, devid});                                                         \
  }                                                                                              \
  void resize_container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn(                               \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat, Ti ni, Ti nj) {   \
    spmat->resize(ni, nj);                                                                       \
  }                                                                                              \
  void resize_container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn##_##virtual(                   \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>> * spmat, Ti ni, Ti nj) {   \
    spmat->resize(ni, nj);                                                                       \
  }                                                                                              \
  void reset_container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn(                                \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat) {                 \
    spmat->_vals.reset(0);                                                                       \
  }                                                                                              \
  void reset_container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn##_##virtual(                    \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>> * spmat) {                 \
    spmat->_vals.reset(0);                                                                       \
  }                                                                                              \
  /* pyview */                                                                                   \
  zs::SpmatViewLite<T, RowMajor, Ti, Tn> *pyview##__##spm##_##T##_##RowMajor##_##Ti##_##Tn(      \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat) {                 \
    return new zs::SpmatViewLite<T, RowMajor, Ti, Tn>{spmat->rows(), spmat->cols(),              \
                                                      spmat->_ptrs.data(), spmat->_inds.data(),  \
                                                      spmat->_vals.data()};                      \
  }                                                                                              \
  zs::SpmatViewLite<const T, RowMajor, Ti, Tn>                                                   \
      *pyview##__##spm##_##const##_##T##_##RowMajor##_##Ti##_##Tn(                               \
          const zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat) {       \
    return new zs::SpmatViewLite<const T, RowMajor, Ti, Tn>{                                     \
        spmat->rows(), spmat->cols(), spmat->_ptrs.data(), spmat->_inds.data(),                  \
        spmat->_vals.data()};                                                                    \
  }                                                                                              \
  zs::SpmatViewLite<T, RowMajor, Ti, Tn>                                                         \
      *pyview##__##spm##_##T##_##RowMajor##_##Ti##_##Tn##_##virtual(                             \
          zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>> * spmat) {             \
    return new zs::SpmatViewLite<T, RowMajor, Ti, Tn>{spmat->rows(), spmat->cols(),              \
                                                      spmat->_ptrs.data(), spmat->_inds.data(),  \
                                                      spmat->_vals.data()};                      \
  }                                                                                              \
  zs::SpmatViewLite<const T, RowMajor, Ti, Tn>                                                   \
      *pyview##__##spm##_##const##_##T##_##RowMajor##_##Ti##_##Tn##_##virtual(                   \
          const zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>> *spmat) {        \
    return new zs::SpmatViewLite<const T, RowMajor, Ti, Tn>{                                     \
        spmat->rows(), spmat->cols(), spmat->_ptrs.data(), spmat->_inds.data(),                  \
        spmat->_vals.data()};                                                                    \
  }                                                                                              \
  void del_pyview##__##spm##_##T##_##RowMajor##_##Ti##_##Tn(                                     \
      zs::SpmatViewLite<T, RowMajor, Ti, Tn> *spmat) {                                           \
    delete spmat;                                                                                \
  }                                                                                              \
  void del_pyview##__##spm##_##const##_##T##_##RowMajor##_##Ti##_##Tn(                           \
      zs::SpmatViewLite<const T, RowMajor, Ti, Tn> *spmat) {                                     \
    delete spmat;                                                                                \
  }

INSTANTIATE_SPMAT_CAPIS(float, false, int, unsigned)
INSTANTIATE_SPMAT_CAPIS(float, true, int, unsigned)
INSTANTIATE_SPMAT_CAPIS(double, false, int, unsigned)
INSTANTIATE_SPMAT_CAPIS(double, true, int, unsigned)

}  // namespace zs
