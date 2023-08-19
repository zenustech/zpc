#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/math/matrix/SparseMatrix.hpp"
#include "zensim/py_interop/SmallVec.hpp"

extern "C" {

#define INSTANTIATE_SPMAT_CAPIS(T, RowMajor, Ti, Tn)                                               \
  void build_from_triplets##__##cuda##_##spm##_##T##_##RowMajor##_##Ti##_##Tn(                     \
      zs::CudaExecutionPolicy *ppol,                                                               \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat, Ti nrows, Ti ncols, \
      Ti *is, Ti *js, T *vals, Tn nnz) {                                                           \
    spmat->build(*ppol, nrows, ncols, zs::range(is, is + nnz), zs::range(js, js + nnz),            \
                 zs::range(vals, vals + nnz));                                                     \
  }                                                                                                \
  void build_from_doublets##__##cuda##_##spm##_##T##_##RowMajor##_##Ti##_##Tn(                     \
      zs::CudaExecutionPolicy *ppol,                                                               \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat, Ti nrows, Ti ncols, \
      Ti *is, Ti *js, Tn nnz) {                                                                    \
    spmat->build(*ppol, nrows, ncols, zs::range(is, is + nnz), zs::range(js, js + nnz),            \
                 zs::false_c);                                                                     \
  }                                                                                                \
  void build_from_doublets_sym##__##cuda##_##spm##_##T##_##RowMajor##_##Ti##_##Tn(                 \
      zs::CudaExecutionPolicy *ppol,                                                               \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat, Ti nrows, Ti ncols, \
      Ti *is, Ti *js, Tn nnz) {                                                                    \
    spmat->build(*ppol, nrows, ncols, zs::range(is, is + nnz), zs::range(js, js + nnz),            \
                 zs::true_c);                                                                      \
  }                                                                                                \
  void fast_build_from_doublets##__##cuda##_##spm##_##T##_##RowMajor##_##Ti##_##Tn(                \
      zs::CudaExecutionPolicy *ppol,                                                               \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat, Ti nrows, Ti ncols, \
      Ti *is, Ti *js, Tn nnz) {                                                                    \
    spmat->fastBuild(*ppol, nrows, ncols, zs::range(is, is + nnz), zs::range(js, js + nnz),        \
                     zs::false_c);                                                                 \
  }                                                                                                \
  void fast_build_from_doublets_sym##__##cuda##_##spm##_##T##_##RowMajor##_##Ti##_##Tn(            \
      zs::CudaExecutionPolicy *ppol,                                                               \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat, Ti nrows, Ti ncols, \
      Ti *is, Ti *js, Tn nnz) {                                                                    \
    spmat->fastBuild(*ppol, nrows, ncols, zs::range(is, is + nnz), zs::range(js, js + nnz),        \
                     zs::true_c);                                                                  \
  }                                                                                                \
  void transpose##__##cuda##_##spm##_##T##_##RowMajor##_##Ti##_##Tn(                               \
      zs::CudaExecutionPolicy *ppol,                                                               \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat) {                   \
    spmat->transposeFrom(*ppol, *spmat, zs::true_c);                                               \
  }

INSTANTIATE_SPMAT_CAPIS(float, false, int, unsigned)
INSTANTIATE_SPMAT_CAPIS(float, true, int, unsigned)
INSTANTIATE_SPMAT_CAPIS(double, false, int, unsigned)
INSTANTIATE_SPMAT_CAPIS(double, true, int, unsigned)
}