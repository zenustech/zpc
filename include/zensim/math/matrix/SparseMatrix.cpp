#include "SparseMatrix.hpp"

namespace zs {

#define INSTANTIATE_SPARSE_MATRIX(T, Ti, Tn)                            \
  ZPC_INSTANTIATE_STRUCT SparseMatrix<T, false, Ti, Tn, ZSPmrAllocator<>>;     \
  ZPC_INSTANTIATE_STRUCT SparseMatrix<T, true, Ti, Tn, ZSPmrAllocator<>>;     \
  ZPC_INSTANTIATE_STRUCT SparseMatrix<T, false, Ti, Tn, ZSPmrAllocator<true>>;  \
  ZPC_INSTANTIATE_STRUCT SparseMatrix<T, true, Ti, Tn, ZSPmrAllocator<true>>;


  INSTANTIATE_SPARSE_MATRIX(f32, i32, i32)
  INSTANTIATE_SPARSE_MATRIX(f32, i32, i64)
  INSTANTIATE_SPARSE_MATRIX(f64, i32, i32)
  INSTANTIATE_SPARSE_MATRIX(f64, i32, i64)

}  // namespace zs