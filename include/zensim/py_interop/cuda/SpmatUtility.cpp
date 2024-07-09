#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/math/matrix/SparseMatrix.hpp"
#include "zensim/py_interop/SmallVec.hpp"

extern "C" {

#define INSTANTIATE_SPMAT_CAPIS(T, RowMajor, Ti, Tn)                                               \
  ZPC_EXPORT void build_from_triplets##__##cuda##_##spm##_##T##_##RowMajor##_##Ti##_##Tn(          \
      zs::CudaExecutionPolicy *ppol,                                                               \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat, Ti nrows, Ti ncols, \
      Ti *is, Ti *js, T *vals, Tn nnz) {                                                           \
    spmat->build(*ppol, nrows, ncols, zs::range(is, is + nnz), zs::range(js, js + nnz),            \
                 zs::range(vals, vals + nnz));                                                     \
  }                                                                                                \
  ZPC_EXPORT void build_from_doublets##__##cuda##_##spm##_##T##_##RowMajor##_##Ti##_##Tn(          \
      zs::CudaExecutionPolicy *ppol,                                                               \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat, Ti nrows, Ti ncols, \
      Ti *is, Ti *js, Tn nnz) {                                                                    \
    spmat->build(*ppol, nrows, ncols, zs::range(is, is + nnz), zs::range(js, js + nnz),            \
                 zs::false_c);                                                                     \
  }                                                                                                \
  ZPC_EXPORT void build_from_doublets_sym##__##cuda##_##spm##_##T##_##RowMajor##_##Ti##_##Tn(      \
      zs::CudaExecutionPolicy *ppol,                                                               \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat, Ti nrows, Ti ncols, \
      Ti *is, Ti *js, Tn nnz) {                                                                    \
    spmat->build(*ppol, nrows, ncols, zs::range(is, is + nnz), zs::range(js, js + nnz),            \
                 zs::true_c);                                                                      \
  }                                                                                                \
  ZPC_EXPORT void fast_build_from_doublets##__##cuda##_##spm##_##T##_##RowMajor##_##Ti##_##Tn(     \
      zs::CudaExecutionPolicy *ppol,                                                               \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat, Ti nrows, Ti ncols, \
      Ti *is, Ti *js, Tn nnz) {                                                                    \
    spmat->fastBuild(*ppol, nrows, ncols, zs::range(is, is + nnz), zs::range(js, js + nnz),        \
                     zs::false_c);                                                                 \
  }                                                                                                \
  ZPC_EXPORT void fast_build_from_doublets_sym##__##cuda##_##spm##_##T##_##RowMajor##_##Ti##_##Tn( \
      zs::CudaExecutionPolicy *ppol,                                                               \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat, Ti nrows, Ti ncols, \
      Ti *is, Ti *js, Tn nnz) {                                                                    \
    spmat->fastBuild(*ppol, nrows, ncols, zs::range(is, is + nnz), zs::range(js, js + nnz),        \
                     zs::true_c);                                                                  \
  }                                                                                                \
  ZPC_EXPORT void transpose##__##cuda##_##spm##_##T##_##RowMajor##_##Ti##_##Tn(                    \
      zs::CudaExecutionPolicy *ppol,                                                               \
      zs::SparseMatrix<T, RowMajor, Ti, Tn, zs::ZSPmrAllocator<false>> *spmat) {                   \
    spmat->transposeFrom(*ppol, *spmat, zs::true_c);                                               \
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
}