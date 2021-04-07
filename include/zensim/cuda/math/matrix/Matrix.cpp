#include "Matrix.hpp"

#include "zensim/Logger.hpp"
#include "zensim/cuda/profile/CudaTimers.cuh"
#include "zensim/tpls/fmt/color.h"

namespace zs {

  template <typename V, typename I> void CudaYaleSparseMatrix<V, I>::analyze_pattern(
      const CudaLibComponentExecutionPolicy<culib_cusolversp> &pol) {
    assert_with_msg(!this->isRowMajor(), "cusparse matrix cannot handle csc format for now!");
    CudaTimer timer{
        Cuda::ref_cuda_context(pol.cupol.getProcid()).stream_spare(pol.cupol.getStreamid())};
    std::size_t sizeInternal, sizeChol;
    timer.tick();
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
    timer.tock("[gpu] analyze pattern");
    auxCholBuffer.resize(sizeChol);
    getchar();
  }
  template <typename V, typename I> void CudaYaleSparseMatrix<V, I>::factorize(
      const CudaLibComponentExecutionPolicy<culib_cusolversp> &pol) {
    assert_with_msg(!this->isRowMajor(), "cusparse matrix cannot handle csc format for now!");
    int singularity{-2};
    CudaTimer timer{
        Cuda::ref_cuda_context(pol.cupol.getProcid()).stream_spare(pol.cupol.getStreamid())};
    if constexpr (is_same_v<V, double>) {
      timer.tick();
      pol.call(cusolverSpDcsrcholFactor, this->rows(), this->nnz(), this->descr, this->vals.data(),
               this->offsets.data(), this->indices.data(), cholInfo, auxCholBuffer.data());
      timer.tock("[gpu] cholesky factorization, A = L*L^T");
      getchar();
      pol.call(cusolverSpDcsrcholZeroPivot, cholInfo, 1e-8, &singularity);
    } else if constexpr (is_same_v<V, float>) {
      timer.tick();
      pol.call(cusolverSpScsrcholFactor, this->rows(), this->nnz(), this->descr, this->vals.data(),
               this->offsets.data(), this->indices.data(), cholInfo, auxCholBuffer.data());
      timer.tock("[gpu] cholesky factorization, A = L*L^T");
      pol.call(cusolverSpScsrcholZeroPivot, cholInfo, 1e-6, &singularity);
    }
    if (0 <= singularity) {
      fmt::print(fg(fmt::color::yellow), "error [gpu] A is not invertible, singularity={}\n",
                 singularity);
      getchar();
    }
  }
  template <typename V, typename I> void CudaYaleSparseMatrix<V, I>::solve(
      zs::Vector<V> &x, const CudaLibComponentExecutionPolicy<culib_cusolversp> &pol,
      const zs::Vector<V> &rhs) {
    assert_with_msg(!this->isRowMajor(), "cusparse matrix cannot handle csc format for now!");
    CudaTimer timer{
        Cuda::ref_cuda_context(pol.cupol.getProcid()).stream_spare(pol.cupol.getStreamid())};
    if constexpr (is_same_v<V, double>) {
      timer.tick();
      pol.call(cusolverSpDcsrcholSolve, this->rows(), rhs.data(), x.data(), cholInfo,
               auxCholBuffer.data());
      timer.tock("[gpu] system solve using cholesky factorization info");
    } else if constexpr (is_same_v<V, float>) {
      timer.tick();
      pol.call(cusolverSpScsrcholSolve, this->rows(), rhs.data(), x.data(), cholInfo,
               auxCholBuffer.data());
      timer.tock("[gpu] system solve using cholesky factorization info");
    }
  }

  template struct CudaYaleSparseMatrix<f32, i32>;
  template struct CudaYaleSparseMatrix<f64, i32>;
}  // namespace zs