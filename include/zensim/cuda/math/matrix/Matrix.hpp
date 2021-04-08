#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolver_common.h>
#include <cusparse_v2.h>

#include "zensim/cuda/execution/CudaLibExecutionPolicy.cuh"
#include "zensim/math/matrix/Matrix.hpp"
#include "zensim/types/Event.hpp"

namespace zs {

  template <typename ValueType, typename IndexType> struct CudaYaleSparseMatrix
      : YaleSparseMatrix<ValueType, IndexType>,
        MatrixAccessor<CudaYaleSparseMatrix<ValueType, IndexType>> {
    using base_t = YaleSparseMatrix<ValueType, IndexType>;
    using value_type = ValueType;
    using index_type = IndexType;
    using event_type = Event<cusparseMatDescr_t, csrcholInfo_t>;
    using listener_type = Event<cusparseMatDescr_t, csrcholInfo_t>;

    template <typename ExecPol>
    CudaYaleSparseMatrix(const YaleSparseMatrix<value_type, index_type> &mat, ExecPol &pol) noexcept
        : YaleSparseMatrix<value_type, index_type>{mat}, auxCholBuffer{zs::memsrc_e::um, 0} {
      pol.template call<culib_cusparse>(cusparseCreateMatDescr, &descr);
      pol.template call<culib_cusparse>(cusparseSetMatType, descr, CUSPARSE_MATRIX_TYPE_GENERAL);
      pol.template call<culib_cusparse>(cusparseSetMatIndexBase, descr, CUSPARSE_INDEX_BASE_ZERO);
      pol.template call<culib_cusolversp>(cusolverSpCreateCsrcholInfo, &cholInfo);
      pol.addListener(
          dtorEvent.createListener([&pol](cusparseMatDescr_t descr, csrcholInfo_t cholInfo) {
            pol.template call<culib_cusparse>(cusparseDestroyMatDescr, descr);
            pol.template call<culib_cusolversp>(cusolverSpDestroyCsrcholInfo, cholInfo);
          }));
    }
    ~CudaYaleSparseMatrix() noexcept { dtorEvent.emit(descr, cholInfo); }

    cusparseMatDescr_t descr{0};
    cusparseSpMatDescr_t spmDescr{0};
    csrcholInfo_t cholInfo{nullptr};

    void analyze_pattern(const CudaLibComponentExecutionPolicy<culib_cusolversp> &pol);
    void factorize(const CudaLibComponentExecutionPolicy<culib_cusolversp> &pol);
    void solve(Vector<value_type> &, const CudaLibComponentExecutionPolicy<culib_cusolversp> &pol,
               const Vector<value_type> &);

    // Vector<char> auxSpmBuffer{};
    Vector<char> auxCholBuffer{};

    event_type dtorEvent;
  };

}  // namespace zs