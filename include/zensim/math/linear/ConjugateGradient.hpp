#pragma once
#include <cmath>

#include "LinearOperators.hpp"
#include "zensim/resource/Resource.h"

namespace zs {

  /// Bow/Math/LinearSolver/ConjugateGradient.h
  template <typename T, int dim, typename Index = std::size_t> struct ConjugateGradient {
    using TV = Vector<T, Index>;
    using allocator_type = ZSPmrAllocator<>;
    using size_type = std::make_unsigned_t<Index>;

    int maxIters;
    TV x_, r_, p_, q_, temp_;
    TV mr_, s_;
    // for dot
    TV dofSqr_;
    TV normSqr_;
    size_type numDofs;
    T tol;
    T relTol;

    ConjugateGradient(const allocator_type& allocator, size_type ndofs)
        : x_{allocator, ndofs},
          r_{allocator, ndofs},
          p_{allocator, ndofs},
          q_{allocator, ndofs},
          temp_{allocator, ndofs},
          mr_{allocator, ndofs},
          s_{allocator, ndofs},
          dofSqr_{allocator, ndofs},
          normSqr_{allocator, 1},
          numDofs{ndofs},
          tol{is_same_v<T, float> ? (T)1e-6 : (T)1e-12},
          maxIters{100},
          relTol{1} {}
    ConjugateGradient(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : ConjugateGradient{get_memory_source(mre, devid), (size_type)0} {}
    ConjugateGradient(size_type count, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : ConjugateGradient{get_memory_source(mre, devid), count} {}

    void resize(size_type ndofs) {
      numDofs = ndofs;
      x_.resize(ndofs);
      r_.resize(ndofs);
      p_.resize(ndofs);
      q_.resize(ndofs);
      temp_.resize(ndofs);
      mr_.resize(ndofs);
      s_.resize(ndofs);
      dofSqr_.resize(ndofs);
    }

    template <typename DV> void print(DV&& dv) {
      for (std::size_t i = 0; i != dv.size(); ++i) fmt::print("{} ", dv.get(i));
      fmt::print("\n");
    }

    template <class ExecutionPolicy, typename DofViewA, typename DofViewB>
    T dotProduct(ExecutionPolicy&& policy, DofViewA a, DofViewB b) {
      constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
      using ValueT = typename std::iterator_traits<RM_CVREF_T(std::begin(a))>::value_type;
      auto dofSqr = dof_view<space, dim>(dofSqr_);
      DofCompwiseOp{std::multiplies<void>{}}(policy, a, b, dofSqr);
      reduce(policy, std::begin(dofSqr), std::end(dofSqr),
             std::begin(dof_view<space, dim>(normSqr_)), 0, std::plus<ValueT>{});
      return normSqr_.clone({memsrc_e::host, -1})[0];
    }

    template <class ExecutionPolicy, typename M, typename XView, typename BView>
    int solve(ExecutionPolicy&& policy, M&& A, XView&& xinout, BView&& b) {
      constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
      resize(xinout.numEntries());
      auto x = dof_view<space, dim>(x_);
      policy(range(numDofs), DofAssign{xinout, x});

      auto r = dof_view<space, dim>(r_), p = dof_view<space, dim>(p_), q = dof_view<space, dim>(q_),
           temp = dof_view<space, dim>(temp_);
      auto mr = dof_view<space, dim>(mr_), s = dof_view<space, dim>(s_);
      auto dofSqr = dof_view<space, dim>(dofSqr_), normSqr = dof_view<space, dim>(normSqr_);

      int iter = 0;
      auto shouldPrint = [](bool v = false) { return v; };
      T alpha, beta, residualPreconditionedNorm, zTrk, zTrkLast;

      A.multiply(policy, x, temp);
      DofCompwiseOp{std::minus<void>{}}(policy, b, temp, r);  // r = b - temp;
      if (shouldPrint()) {
        auto res = dotProduct(policy, r, r);
        fmt::print("(after minus Ax) normSqr rhs: {}\n", res);
      }
      fmt::print("check num dofs: {}, r dofs: {}\n", x.numEntries(), r.numEntries());
      A.project(policy, r);
      fmt::print("done proj, r dofs: {}\n", r.numEntries());
      if (shouldPrint()) {
        auto res = dotProduct(policy, r, r);
        fmt::print("(after proj) normSqr rhs: {}\n", res);
      }

      A.precondition(policy, r, q);  // NOTE: requires that preconditioning matrix is projected
      if (shouldPrint()) {
        auto res = dotProduct(policy, q, q);
        fmt::print("(after precondition) normSqr rhs: {}\n", res);
      }
      fmt::print("done precondition, r dofs: {}\n", q.numEntries());
      policy(range(numDofs), DofAssign{q, p});  // p = q;
      if (shouldPrint()) {
        auto res = dotProduct(policy, p, p);
        fmt::print("(after assign) normSqr rhs: {}\n", res);
      }

      zTrk = dotProduct(policy, r, q);  // zTrk = std::abs(dotProduct(r, q));
      fmt::print("iter: {}, zTrk {}\n", iter, zTrk);
      residualPreconditionedNorm = std::sqrt(zTrk);
      T localTol = std::min(relTol * residualPreconditionedNorm, tol);
      for (; iter != maxIters; ++iter) {
        // if (iter % 10 == 9) getchar();
        if (shouldPrint(iter % 10 == 9))
          fmt::print("iter: {}, norm: {}, tol {}\n", iter, residualPreconditionedNorm, localTol);
        if (residualPreconditionedNorm <= localTol) {
          policy(range(numDofs), DofAssign{x, xinout});
          ///
          // print(xinout);
          ///
          return iter;
        }
        A.multiply(policy, p, temp);
        if (shouldPrint(iter % 10 == 9)) fmt::print("iter: {}, done multiply\n", iter);
        A.project(policy, temp);
        if (shouldPrint(iter % 10 == 9)) fmt::print("iter: {}, done project\n", iter);
        alpha = zTrk / dotProduct(policy, temp, p);  // alpha = zTrk / dotProduct(temp, p);
        if (shouldPrint(iter % 10 == 9)) fmt::print("iter: {}, alpha {}\n", iter, alpha);

        DofCompwiseOp{LinearCombineOp(alpha)}(policy, p, x, x);  // x = x + alpha * p;
        if (shouldPrint(iter % 10 == 9)) fmt::print("iter: {}, done x += a * p\n", iter);
        DofCompwiseOp{LinearCombineOp(-alpha)}(policy, temp, r, r);  // r = r - alpha * temp;
        if (shouldPrint(iter % 10 == 9)) fmt::print("iter: {}, done r -= a * temp\n", iter);
        A.precondition(policy, r, q);  // NOTE: requires that preconditioning matrix is projected
        if (shouldPrint(iter % 10 == 9)) fmt::print("iter: {}, done precondition\n", iter);

        zTrkLast = zTrk;
        if (shouldPrint(iter % 10 == 9)) fmt::print("iter: {}, last ztrk {}\n", iter, zTrkLast);
        zTrk = dotProduct(policy, q, r);  // zTrk = dotProduct(q, r);
        if (shouldPrint(iter % 10 == 9)) fmt::print("iter: {}, new ztrk {}\n", iter, zTrk);
        beta = zTrk / zTrkLast;
        if (shouldPrint(iter % 10 == 9)) fmt::print("iter: {}, beta {}\n", iter, beta);

        DofCompwiseOp{LinearCombineOp(beta)}(policy, p, q, p);  // p = q + beta * p;
        if (shouldPrint(iter % 10 == 9)) fmt::print("iter: {}, done p = q + beta * p\n", iter);

        residualPreconditionedNorm = std::sqrt(zTrk);
      }
      return iter;
    }
  };

}  // namespace zs