#include "zensim/math/Vec.h"
#include "zensim/math/matrix/SparseMatrix.hpp"
#include "zensim/py_interop/SmallVec.hpp"
#include "zensim/py_interop/SpmatView.hpp"

extern "C" {

#define INSTANTIATE_SPMAT_CAPIS(T, RowMajor, Ti, Tn)                                             \
  /* container */                                                                                \
  ZPC_EXPORT zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>>                    \
      *container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn(                                      \
          const zs::ZSPmrAllocator<false> *allocator, Ti ni, Ti nj) {                            \
    return new zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>>{*allocator, ni,  \
                                                                                nj};             \
  }                                                                                              \
  ZPC_EXPORT zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>>                     \
      *container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn##_##virtual(                          \
          const zs::ZSPmrAllocator<true> *allocator, Ti ni, Ti nj) {                             \
    return new zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>>{*allocator, ni,   \
                                                                               nj};              \
  }                                                                                              \
  ZPC_EXPORT void del_container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn(                       \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat) {                 \
    delete spmat;                                                                                \
  }                                                                                              \
  ZPC_EXPORT void del_container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn##_##virtual(           \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>> * spmat) {                 \
    delete spmat;                                                                                \
  }                                                                                              \
  ZPC_EXPORT void relocate_container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn(                  \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat, zs::memsrc_e mre, \
      zs::ProcID devid) {                                                                        \
    *spmat = spmat->clone({mre, devid});                                                         \
  }                                                                                              \
  ZPC_EXPORT void relocate_container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn##_##virtual(      \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>> * spmat, zs::memsrc_e mre, \
      zs::ProcID devid) {                                                                        \
    *spmat = spmat->clone({mre, devid});                                                         \
  }                                                                                              \
  ZPC_EXPORT void resize_container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn(                    \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat, Ti ni, Ti nj) {   \
    spmat->resize(ni, nj);                                                                       \
  }                                                                                              \
  ZPC_EXPORT void resize_container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn##_##virtual(        \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>> * spmat, Ti ni, Ti nj) {   \
    spmat->resize(ni, nj);                                                                       \
  }                                                                                              \
  ZPC_EXPORT void reset_container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn(                     \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat) {                 \
    spmat->_vals.reset(0);                                                                       \
  }                                                                                              \
  ZPC_EXPORT void reset_container##__##spm##_##T##_##RowMajor##_##Ti##_##Tn##_##virtual(         \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>> * spmat) {                 \
    spmat->_vals.reset(0);                                                                       \
  }                                                                                              \
  /* custom */                                                                                   \
  ZPC_EXPORT size_t nnz##__##spm##_##T##_##RowMajor##_##Ti##_##Tn(                               \
      const zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat) {           \
    return spmat->nnz();                                                                         \
  }                                                                                              \
  ZPC_EXPORT size_t nnz##__##spm##_##T##_##RowMajor##_##Ti##_##Tn##_##virtual(                   \
      const zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>> *spmat) {            \
    return spmat->nnz();                                                                         \
  }                                                                                              \
  ZPC_EXPORT size_t outer_size##__##spm##_##T##_##RowMajor##_##Ti##_##Tn(                        \
      const zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat) {           \
    return spmat->outerSize();                                                                   \
  }                                                                                              \
  ZPC_EXPORT size_t outer_size##__##spm##_##T##_##RowMajor##_##Ti##_##Tn##_##virtual(            \
      const zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>> *spmat) {            \
    return spmat->outerSize();                                                                   \
  }                                                                                              \
  ZPC_EXPORT size_t inner_size##__##spm##_##T##_##RowMajor##_##Ti##_##Tn(                        \
      const zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat) {           \
    return spmat->innerSize();                                                                   \
  }                                                                                              \
  ZPC_EXPORT size_t inner_size##__##spm##_##T##_##RowMajor##_##Ti##_##Tn##_##virtual(            \
      const zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>> *spmat) {            \
    return spmat->innerSize();                                                                   \
  }                                                                                              \
  /* pyview */                                                                                   \
  ZPC_EXPORT zs::SpmatViewLite<T, RowMajor, Ti, Tn>                                              \
      *pyview##__##spm##_##T##_##RowMajor##_##Ti##_##Tn(                                         \
          zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat) {             \
    return new zs::SpmatViewLite<T, RowMajor, Ti, Tn>{spmat->rows(), spmat->cols(),              \
                                                      spmat->_ptrs.data(), spmat->_inds.data(),  \
                                                      spmat->_vals.data()};                      \
  }                                                                                              \
  ZPC_EXPORT zs::SpmatViewLite<const T, RowMajor, Ti, Tn>                                        \
      *pyview##__##spm##_##const##_##T##_##RowMajor##_##Ti##_##Tn(                               \
          const zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat) {       \
    return new zs::SpmatViewLite<const T, RowMajor, Ti, Tn>{                                     \
        spmat->rows(), spmat->cols(), spmat->_ptrs.data(), spmat->_inds.data(),                  \
        spmat->_vals.data()};                                                                    \
  }                                                                                              \
  ZPC_EXPORT zs::SpmatViewLite<T, RowMajor, Ti, Tn>                                              \
      *pyview##__##spm##_##T##_##RowMajor##_##Ti##_##Tn##_##virtual(                             \
          zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>> * spmat) {             \
    return new zs::SpmatViewLite<T, RowMajor, Ti, Tn>{spmat->rows(), spmat->cols(),              \
                                                      spmat->_ptrs.data(), spmat->_inds.data(),  \
                                                      spmat->_vals.data()};                      \
  }                                                                                              \
  ZPC_EXPORT zs::SpmatViewLite<const T, RowMajor, Ti, Tn>                                        \
      *pyview##__##spm##_##const##_##T##_##RowMajor##_##Ti##_##Tn##_##virtual(                   \
          const zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<true>> *spmat) {        \
    return new zs::SpmatViewLite<const T, RowMajor, Ti, Tn>{                                     \
        spmat->rows(), spmat->cols(), spmat->_ptrs.data(), spmat->_inds.data(),                  \
        spmat->_vals.data()};                                                                    \
  }                                                                                              \
  ZPC_EXPORT void del_pyview##__##spm##_##T##_##RowMajor##_##Ti##_##Tn(                          \
      zs::SpmatViewLite<T, RowMajor, Ti, Tn> *spmat) {                                           \
    delete spmat;                                                                                \
  }                                                                                              \
  ZPC_EXPORT void del_pyview##__##spm##_##const##_##T##_##RowMajor##_##Ti##_##Tn(                \
      zs::SpmatViewLite<const T, RowMajor, Ti, Tn> *spmat) {                                     \
    delete spmat;                                                                                \
  }

using mat33f = zs::vec<float, 3, 3>;
using mat33d = zs::vec<double, 3, 3>;

INSTANTIATE_SPMAT_CAPIS(float, false, int, unsigned)
INSTANTIATE_SPMAT_CAPIS(float, true, int, unsigned)
INSTANTIATE_SPMAT_CAPIS(double, false, int, unsigned)
INSTANTIATE_SPMAT_CAPIS(double, true, int, unsigned)
INSTANTIATE_SPMAT_CAPIS(mat33f, false, int, unsigned)
INSTANTIATE_SPMAT_CAPIS(mat33f, true, int, unsigned)
INSTANTIATE_SPMAT_CAPIS(mat33d, false, int, unsigned)
INSTANTIATE_SPMAT_CAPIS(mat33d, true, int, unsigned)

}  // namespace zs
