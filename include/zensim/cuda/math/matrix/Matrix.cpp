#include "Matrix.hpp"

#include "zensim/Logger.hpp"

namespace zs {

  template <typename V, typename I> void CudaYaleSparseMatrix<V, I>::analyze_pattern(
      const CudaLibComponentExecutionPolicy<culib_cusolversp> &pol) {
    std::size_t sizeInternal, sizeChol;
    if (this->isRowMajor())
      pol.call(cusolverSpXcsrcholAnalysis, this->rows(), this->nnz(), this->descr,
               this->offsets.data(), this->indices.data(), cholInfo);
    // else
    //   pol.call(cusolverSpXcsccholAnalysis, this->cols(), this->nnz(), this->descr,
    //            this->offsets.data(), this->indices.data(), cholInfo);

    if constexpr (is_same_v<V, double>) {
      if (this->isRowMajor())
        pol.call(cusolverSpDcsrcholBufferInfo, this->rows(), this->nnz(), this->descr,
                 this->vals.data(), this->offsets.data(), this->indices.data(), cholInfo,
                 &sizeInternal, &sizeChol);
      // else
      //   pol.call(cusolverSpDcsccholBufferInfo, this->cols(), this->nnz(), this->descr,
      //            this->vals.data(), this->offsets.data(), this->indices.data(), cholInfo,
      //            &sizeInternal, &sizeChol);
    } else if constexpr (is_same_v<V, float>) {
      if (this->isRowMajor())
        pol.call(cusolverSpScsrcholBufferInfo, this->rows(), this->nnz(), this->descr,
                 this->vals.data(), this->offsets.data(), this->indices.data(), cholInfo,
                 &sizeInternal, &sizeChol);
      // else
      //   pol.call(cusolverSpScsccholBufferInfo, this->cols(), this->nnz(), this->descr,
      //            this->vals.data(), this->offsets.data(), this->indices.data(), cholInfo,
      //            &sizeInternal, &sizeChol);
    }
    auxCholBuffer.resize(sizeChol);
  }
  template <typename V, typename I> void CudaYaleSparseMatrix<V, I>::factorize(
      const CudaLibComponentExecutionPolicy<culib_cusolversp> &pol) {
    ;
    ;
  }
  template <typename V, typename I> void CudaYaleSparseMatrix<V, I>::solve(
      const CudaLibComponentExecutionPolicy<culib_cusolversp> &pol) {
    ;
    ;
  }

  template struct CudaYaleSparseMatrix<f32, i32>;
  template struct CudaYaleSparseMatrix<f64, i32>;
}  // namespace zs