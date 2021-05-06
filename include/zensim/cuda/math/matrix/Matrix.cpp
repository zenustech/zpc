#include "Matrix.hpp"

#include "zensim/Logger.hpp"
#include "zensim/cuda/profile/CudaTimers.cuh"
#include "zensim/tpls/fmt/color.h"

namespace zs {

  template <typename V, typename I> void CudaYaleSparseMatrix<V, I>::analyze_pattern(
      const CudaLibComponentExecutionPolicy<culib_cusolversp> &pol) {
    assert_with_msg(!this->isRowMajor(), "cusparse matrix cannot handle csc format for now!");
    CudaTimer timer{
        Cuda::ref_cuda_context(pol.get().getProcid()).streamSpare(pol.get().getStreamid())};
    std::size_t sizeInternal, sizeChol;
    timer.tick();
    if (this->isRowMajor())
      pol.call(cusolverSpXcsrcholAnalysis, this->rows(), this->nnz(), this->descr,
               this->offsets.data(), this->indices.data(), cholInfo);
    timer.tock("[gpu] analyze pattern");
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
    assert_with_msg(!this->isRowMajor(), "cusparse matrix cannot handle csc format for now!");
    int singularity{-2};
    CudaTimer timer{
        Cuda::ref_cuda_context(pol.get().getProcid()).streamSpare(pol.get().getStreamid())};
    if constexpr (is_same_v<V, double>) {
      timer.tick();
      pol.call(cusolverSpDcsrcholFactor, this->rows(), this->nnz(), this->descr, this->vals.data(),
               this->offsets.data(), this->indices.data(), cholInfo, auxCholBuffer.data());
      timer.tock("[gpu] cholesky factorization, A = L*L^T");
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
    }
  }
  template <typename V, typename I> void CudaYaleSparseMatrix<V, I>::solve(
      zs::Vector<V> &x, const CudaLibComponentExecutionPolicy<culib_cusolversp> &pol,
      const zs::Vector<V> &rhs) {
    assert_with_msg(!this->isRowMajor(), "cusparse matrix cannot handle csc format for now!");
    CudaTimer timer{
        Cuda::ref_cuda_context(pol.get().getProcid()).streamSpare(pol.get().getStreamid())};
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
  template <typename V, typename I> void CudaYaleSparseMatrix<V, I>::cgsolve(
      zs::Vector<V> &x,
      const CudaLibExecutionPolicy<culib_cusparse, culib_cublas, culib_cusolversp> &pol,
      const zs::Vector<V> &rhs) {
    assert_with_msg(!this->isRowMajor(), "cusparse matrix cannot handle csc format for now!");
    CudaTimer timer{
        Cuda::ref_cuda_context(pol.get().getProcid()).streamSpare(pol.get().getStreamid())};

    const auto &sparse = static_cast<const CudaLibComponentExecutionPolicy<culib_cusparse> &>(pol);
    const auto &blas = static_cast<const CudaLibComponentExecutionPolicy<culib_cublas> &>(pol);
    const auto &solversp
        = static_cast<const CudaLibComponentExecutionPolicy<culib_cusolversp> &>(pol);
    if constexpr (is_same_v<V, double>) {
      timer.tick();

      double r1;

      double alpha = 1.0;
      double alpham1 = -1.0;
      double beta = 0.0;
      double r0 = 0.;
      auto r = rhs, p = x, Ax = x;
      sparse.call(cusparseCreateCsr, &spmDescr, this->rows(), this->cols(), this->nnz(),
                  this->offsets.data(), this->indices.data(), this->vals.data(), CUSPARSE_INDEX_32I,
                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
      cusparseDnVecDescr_t vecx = NULL;
      sparse.call(cusparseCreateDnVec, &vecx, this->rows(), x.data(), CUDA_R_64F);
      cusparseDnVecDescr_t vecp = NULL;
      sparse.call(cusparseCreateDnVec, &vecp, this->rows(), p.data(), CUDA_R_64F);
      cusparseDnVecDescr_t vecAx = NULL;
      sparse.call(cusparseCreateDnVec, &vecAx, this->rows(), Ax.data(), CUDA_R_64F);

      /* Allocate workspace for cuSPARSE */
      size_t bufferSize = 0;
      sparse.call(cusparseSpMV_bufferSize, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spmDescr, vecx,
                  &beta, vecAx, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
      auxCholBuffer.resize(bufferSize);

      sparse.call(cusparseSpMV, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spmDescr, vecx, &beta,
                  vecAx, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, auxCholBuffer.data());

      blas.call(cublasDaxpy, this->rows(), &alpham1, Ax.data(), 1, r.data(), 1);
      blas.call(cublasDdot, this->rows(), r.data(), 1, r.data(), 1, &r1);

      int k = 1;

#if 0
      while (r1 > tol * tol && k <= max_iter) {
        if (k > 1) {
          b = r1 / r0;
          cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);
          cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        } else {
          cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
        }

        checkCudaErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                                     vecp, &beta, vecAx, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT,
                                     &buffer));
        cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
        a = r1 / dot;

        cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
        na = -a;
        cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

        r0 = r1;
        cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        cudaDeviceSynchronize();
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
      }
#endif

      timer.tock("[gpu] system conjugate gradient solve");
    } else if constexpr (is_same_v<V, float>) {
      timer.tick();

      timer.tock("[gpu] system conjugate gradient solve");
    }
  }

  template struct CudaYaleSparseMatrix<f32, i32>;
  template struct CudaYaleSparseMatrix<f64, i32>;
}  // namespace zs