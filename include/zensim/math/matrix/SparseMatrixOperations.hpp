#pragma once
#include "SparseMatrix.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  ///@note spmv row major
  template <typename Policy, typename T, typename Ti, typename Tn, typename AllocatorT,
            typename InVRangeT, typename OutVRangeT>
  inline void spmv_classic(Policy &&policy, const SparseMatrix<T, true, Ti, Tn, AllocatorT> &spmat,
                           InVRangeT &&inV, OutVRangeT &&outV) {
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;

    auto nrows = spmat.rows();
    auto ncols = spmat.cols();
    if (range_size(inV) != ncols || range_size(outV) != nrows)
      throw std::runtime_error("spmv size mismatch");
    if (!valid_memspace_for_execution(policy, spmat.get_allocator()))
      throw std::runtime_error("current memory location not compatible with the execution policy");
    /// @note compilation checks
    // assert_backend_presence<space>();

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
                 atomic_add(execTag, &vout[row].val(d), (typename TOut::value_type)sum.val(d));
             else
               atomic_add(execTag, &vout[row], (TOut)sum);
           });
  }

  ///@note spmv col major
  template <typename Policy, typename T, typename Ti, typename Tn, typename AllocatorT,
            typename InVRangeT, typename OutVRangeT>
  inline void spmv_classic(Policy &&policy, const SparseMatrix<T, false, Ti, Tn, AllocatorT> &spmat,
                           InVRangeT &&inV, OutVRangeT &&outV) {
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;

    auto nrows = spmat.rows();
    auto ncols = spmat.cols();
    if (range_size(inV) != ncols || range_size(outV) != nrows)
      throw std::runtime_error("spmv size mismatch");
    if (!valid_memspace_for_execution(policy, spmat.get_allocator()))
      throw std::runtime_error("current memory location not compatible with the execution policy");
    /// @note compilation checks
    // assert_backend_presence<space>();

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
                   atomic_add(execTag, &vout[row].val(d), (typename TOut::value_type)delta.val(d));
               else
                 atomic_add(execTag, &vout[row], (TOut)delta);
             }
           });
  }

  ///@note spmv (semiring) row major
  template <typename Policy, typename T, typename Ti, typename Tn, typename AllocatorT,
            typename InVRangeT, typename OutVRangeT, semiring_e category = semiring_e::plus_times>
  inline void spmv(Policy &&policy, const SparseMatrix<T, true, Ti, Tn, AllocatorT> &spmat,
                   InVRangeT &&inV, OutVRangeT &&outV, wrapv<category> = {}) {
    using TOut = RM_CVREF_T(*std::begin(outV));
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    constexpr auto sr = make_semiring(wrapv<category>{}, wrapt<TOut>{});

    auto nrows = spmat.rows();
    auto ncols = spmat.cols();
    if (range_size(inV) != ncols || range_size(outV) != nrows)
      throw std::runtime_error("spmv size mismatch");
    if (!valid_memspace_for_execution(policy, spmat.get_allocator()))
      throw std::runtime_error("current memory location not compatible with the execution policy");
    /// @note compilation checks
    // assert_backend_presence<space>();

    static_assert(std::is_convertible_v<T, TOut>, "output type incompatible with spmat value_type");
    policy(range(nrows), [vout = std::begin(outV), sr = sr] ZS_LAMBDA(Ti row) mutable {
      vout[row] = sr.identity();
    });
    policy(range(nrows),
           [spmat = proxy<space>(spmat), vin = std::begin(inV), vout = std::begin(outV),
            execTag = wrapv<space>{}, sr = sr] ZS_LAMBDA(Ti row) mutable {
             auto bg = spmat._ptrs[row];
             auto ed = spmat._ptrs[row + 1];
             auto sum = sr.identity();
             for (auto i = bg; i < ed; ++i)
               sum = sr.add(sum, sr.multiply(spmat._vals[i], vin[spmat._inds[i]]));

             using monoid_type = typename RM_CVREF_T(sr)::monoid_type;
             if constexpr (is_same_v<monoid_type, monoid<plus<TOut>>>)
               atomic_add(execTag, &vout[row], (TOut)sum);
             else if constexpr (is_same_v<monoid_type, monoid<logical_or<TOut>>>)
               atomic_or(execTag, &vout[row], (TOut)sum);
             else if constexpr (is_same_v<monoid_type, monoid<getmin<TOut>>>)
               atomic_min(execTag, &vout[row], (TOut)sum);
             else if constexpr (is_same_v<monoid_type, monoid<getmax<TOut>>>)
               atomic_max(execTag, &vout[row], (TOut)sum);
           });
  }

  ///@note spmv (semiring) col major
  template <typename Policy, typename T, typename Ti, typename Tn, typename AllocatorT,
            typename InVRangeT, typename OutVRangeT, semiring_e category = semiring_e::plus_times>
  inline void spmv(Policy &&policy, const SparseMatrix<T, false, Ti, Tn, AllocatorT> &spmat,
                   InVRangeT &&inV, OutVRangeT &&outV, wrapv<category> = {}) {
    using TOut = RM_CVREF_T(*std::begin(outV));
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    constexpr auto sr = make_semiring(wrapv<category>{}, wrapt<TOut>{});

    auto nrows = spmat.rows();
    auto ncols = spmat.cols();
    if (range_size(inV) != ncols || range_size(outV) != nrows)
      throw std::runtime_error("spmv size mismatch");
    if (!valid_memspace_for_execution(policy, spmat.get_allocator()))
      throw std::runtime_error("current memory location not compatible with the execution policy");
    /// @note compilation checks
    // assert_backend_presence<space>();

    static_assert(std::is_convertible_v<T, TOut>, "output type incompatible with spmat value_type");
    policy(range(nrows), [vout = std::begin(outV), sr = sr] ZS_LAMBDA(Ti row) mutable {
      vout[row] = sr.identity();
    });
    policy(range(ncols),
           [spmat = proxy<space>(spmat), vin = std::begin(inV), vout = std::begin(outV),
            execTag = wrapv<space>{}, sr = sr] ZS_LAMBDA(Ti col) mutable {
             auto bg = spmat._ptrs[col];
             auto ed = spmat._ptrs[col + 1];
             auto v = vin[col];
             for (auto i = bg; i < ed; ++i) {
               auto delta = sr.multiply(spmat._vals[i], v);
               auto row = spmat._inds[i];

               using monoid_type = typename RM_CVREF_T(sr)::monoid_type;
               if constexpr (is_same_v<monoid_type, monoid<plus<TOut>>>)
                 atomic_add(execTag, &vout[row], (TOut)delta);
               else if constexpr (is_same_v<monoid_type, monoid<logical_or<TOut>>>)
                 atomic_or(execTag, &vout[row], (TOut)delta);
               else if constexpr (is_same_v<monoid_type, monoid<getmin<TOut>>>)
                 atomic_min(execTag, &vout[row], (TOut)delta);
               else if constexpr (is_same_v<monoid_type, monoid<getmax<TOut>>>)
                 atomic_max(execTag, &vout[row], (TOut)delta);
             }
           });
  }

  /// @brief mask variants (output sparsity)

  ///@note spmv_mask (semiring) row major
  template <typename Policy, typename T, typename Ti, typename Tn, typename AllocatorT,
            typename InVRangeT, typename MaskRangeT, typename OutVRangeT,
            semiring_e category = semiring_e::plus_times>
  inline void spmv_mask(Policy &&policy, const SparseMatrix<T, true, Ti, Tn, AllocatorT> &spmat,
                        InVRangeT &&inV, MaskRangeT &&mask, OutVRangeT &&outV,
                        wrapv<category> = {}) {
    using TOut = RM_CVREF_T(*std::begin(outV));
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    constexpr auto sr = make_semiring(wrapv<category>{}, wrapt<TOut>{});

    auto nrows = spmat.rows();
    auto ncols = spmat.cols();
    if (range_size(inV) != ncols || range_size(outV) != nrows)
      throw std::runtime_error("spmv size mismatch");
    if (!valid_memspace_for_execution(policy, spmat.get_allocator()))
      throw std::runtime_error("current memory location not compatible with the execution policy");
    /// @note compilation checks
    // assert_backend_presence<space>();

    static_assert(std::is_convertible_v<T, TOut>, "output type incompatible with spmat value_type");
    policy(range(nrows), [vout = std::begin(outV), sr = sr] ZS_LAMBDA(Ti row) mutable {
      vout[row] = sr.identity();
    });
    policy(range(nrows),
           [spmat = proxy<space>(spmat), vin = std::begin(inV), mask = std::begin(mask),
            vout = std::begin(outV), execTag = wrapv<space>{}, sr = sr] ZS_LAMBDA(Ti row) mutable {
             if (mask[row]) return;

             auto bg = spmat._ptrs[row];
             auto ed = spmat._ptrs[row + 1];
             auto sum = sr.identity();
             for (auto i = bg; i < ed; ++i)
               sum = sr.add(sum, sr.multiply(spmat._vals[i], vin[spmat._inds[i]]));

             using monoid_type = typename RM_CVREF_T(sr)::monoid_type;
             if constexpr (is_same_v<monoid_type, monoid<plus<TOut>>>)
               atomic_add(execTag, &vout[row], (TOut)sum);
             else if constexpr (is_same_v<monoid_type, monoid<logical_or<TOut>>>)
               atomic_or(execTag, &vout[row], (TOut)sum);
             else if constexpr (is_same_v<monoid_type, monoid<getmin<TOut>>>)
               atomic_min(execTag, &vout[row], (TOut)sum);
             else if constexpr (is_same_v<monoid_type, monoid<getmax<TOut>>>)
               atomic_max(execTag, &vout[row], (TOut)sum);
           });
  }

  ///@note spmv_mask (semiring) col major
  template <typename Policy, typename T, typename Ti, typename Tn, typename AllocatorT,
            typename InVRangeT, typename MaskRangeT, typename OutVRangeT,
            semiring_e category = semiring_e::plus_times>
  inline void spmv_mask(Policy &&policy, const SparseMatrix<T, false, Ti, Tn, AllocatorT> &spmat,
                        InVRangeT &&inV, MaskRangeT &&mask, OutVRangeT &&outV,
                        wrapv<category> = {}) {
    using TOut = RM_CVREF_T(*std::begin(outV));
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    constexpr auto sr = make_semiring(wrapv<category>{}, wrapt<TOut>{});

    auto nrows = spmat.rows();
    auto ncols = spmat.cols();
    if (range_size(inV) != ncols || range_size(outV) != nrows)
      throw std::runtime_error("spmv size mismatch");
    if (!valid_memspace_for_execution(policy, spmat.get_allocator()))
      throw std::runtime_error("current memory location not compatible with the execution policy");
    /// @note compilation checks
    // assert_backend_presence<space>();

    static_assert(std::is_convertible_v<T, TOut>, "output type incompatible with spmat value_type");
    policy(range(nrows), [vout = std::begin(outV), sr = sr] ZS_LAMBDA(Ti row) mutable {
      vout[row] = sr.identity();
    });
    policy(range(ncols),
           [spmat = proxy<space>(spmat), vin = std::begin(inV), mask = std::begin(mask),
            vout = std::begin(outV), execTag = wrapv<space>{}, sr = sr] ZS_LAMBDA(Ti col) mutable {
             auto bg = spmat._ptrs[col];
             auto ed = spmat._ptrs[col + 1];
             auto v = vin[col];
             for (auto i = bg; i < ed; ++i) {
               auto row = spmat._inds[i];
               if (mask[row]) continue;

               auto delta = sr.multiply(spmat._vals[i], v);
               using monoid_type = typename RM_CVREF_T(sr)::monoid_type;
               if constexpr (is_same_v<monoid_type, monoid<plus<TOut>>>)
                 atomic_add(execTag, &vout[row], (TOut)delta);
               else if constexpr (is_same_v<monoid_type, monoid<logical_or<TOut>>>)
                 atomic_or(execTag, &vout[row], (TOut)delta);
               else if constexpr (is_same_v<monoid_type, monoid<getmin<TOut>>>)
                 atomic_min(execTag, &vout[row], (TOut)delta);
               else if constexpr (is_same_v<monoid_type, monoid<getmax<TOut>>>)
                 atomic_max(execTag, &vout[row], (TOut)delta);
             }
           });
  }

}  // namespace zs