#pragma once
#include "SparseMatrix.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  ///@note row major
  template <typename Policy, typename T, typename Ti, typename Tn, typename AllocatorT,
            typename InVRangeT, typename OutVRangeT>
  inline void SpMV(Policy &&policy, const SparseMatrix<T, true, Ti, Tn, AllocatorT> &spmat,
                   InVRangeT &&inV, OutVRangeT &&outV) {
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;

    auto nrows = spmat.rows();
    auto ncols = spmat.cols();
    if (range_size(inV) != ncols || range_size(outV) != nrows)
      throw std::runtime_error("spmv size mismatch");
    if (!valid_memspace_for_execution(policy, spmat.get_allocator()))
      throw std::runtime_error("current memory location not compatible with the execution policy");
    /// @note compilation checks
    assert_backend_presence<space>();

    using TOut = RM_CVREF_T(*std::begin(outV));
    policy(range(nrows),
           [spmat = proxy<space>(spmat), vin = std::begin(inV), vout = std::begin(outV),
            execTag = wrapv<space>{}] ZS_LAMBDA(Ti row) mutable {
             auto bg = spmat._ptrs[row];
             auto ed = spmat._ptrs[row + 1];
             T sum = 0;
             for (auto i = bg; i < ed; ++i) sum += spmat._vals[i] * vin[spmat._inds[i]];

             if constexpr (is_vec<TOut>::value)
               for (typename TOut::index_type d = 0; d != TOut::extent; ++d)
                 atomic_add(execTag, &outV[row].val(d), (typename TOut::value_type)sum.val(d));
             else
               atomic_add(execTag, &outV[row], (TOut)sum);
           });
  }

  ///@note col major
  template <typename Policy, typename T, typename Ti, typename Tn, typename AllocatorT,
            typename InVRangeT, typename OutVRangeT>
  inline void SpMV(Policy &&policy, const SparseMatrix<T, false, Ti, Tn, AllocatorT> &spmat,
                   InVRangeT &&inV, OutVRangeT &&outV) {
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;

    auto nrows = spmat.rows();
    auto ncols = spmat.cols();
    if (range_size(inV) != ncols || range_size(outV) != nrows)
      throw std::runtime_error("spmv size mismatch");
    if (!valid_memspace_for_execution(policy, spmat.get_allocator()))
      throw std::runtime_error("current memory location not compatible with the execution policy");
    /// @note compilation checks
    assert_backend_presence<space>();

    using TOut = RM_CVREF_T(*std::begin(outV));
    policy(range(ncols),
           [spmat = proxy<space>(spmat), vin = std::begin(inV), vout = std::begin(outV),
            execTag = wrapv<space>{}] ZS_LAMBDA(Ti col) mutable {
             auto bg = spmat._ptrs[col];
             auto ed = spmat._ptrs[col + 1];
             auto v = vin[col];
             for (auto i = bg; i < ed; ++i) {
               auto delta = spmat._vals[i] * v;
               auto row = spmat._inds[i];
               if constexpr (is_vec<TOut>::value)
                 for (typename TOut::index_type d = 0; d != TOut::extent; ++d)
                   atomic_add(execTag, &outV[row].val(d), (typename TOut::value_type)delta.val(d));
               else
                 atomic_add(execTag, &outV[row], (TOut)delta);
             }
           });
  }

}  // namespace zs