#pragma once
#include <cmath>

#include "LinearOperators.hpp"

namespace zs {

  /// Bow/Math/LinearSolver/ConjugateGradient.h
  template <typename T, int dim, typename Index = zs::size_t> struct ConjugateResidual {
    using TV = Vector<T>;
    using allocator_type = ZSPmrAllocator<>;
    using size_type = zs::make_unsigned_t<Index>;

    int maxIters;
    TV x_, or_, r_, p_, Ap_, Ar_, rAr_, MAp_;
    // for dot
    TV dofSqr_;
    TV normSqr_;
    size_type numDofs;
    T tol;
    T relTol;

    ConjugateResidual(const allocator_type& allocator, size_type ndofs)
        : x_{allocator, ndofs},
          or_{allocator, ndofs},
          r_{allocator, ndofs},
          p_{allocator, ndofs},
          Ap_{allocator, ndofs},
          Ar_{allocator, ndofs},
          MAp_{allocator, ndofs},
          dofSqr_{allocator, ndofs},
          normSqr_{allocator, 1},
          numDofs{ndofs},
          tol{is_same_v<T, float> ? (T)1e-6 : (T)1e-12},
          maxIters{100},
          relTol{1} {}
    ConjugateResidual(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : ConjugateResidual{get_memory_source(mre, devid), (size_type)0} {}
    ConjugateResidual(size_type count, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : ConjugateResidual{get_memory_source(mre, devid), count} {}

    void resize(size_type ndofs) {
      numDofs = ndofs;
      x_.resize(ndofs);
      or_.resize(ndofs);
      r_.resize(ndofs);
      p_.resize(ndofs);
      Ap_.resize(ndofs);
      Ar_.resize(ndofs);
      MAp_.resize(ndofs);
      dofSqr_.resize(ndofs);
    }

    template <typename DV> void print(DV&& dv) {
      for (size_t i = 0; i != dv.size(); ++i) fmt::print("{} ", dv.get(i));
      fmt::print("\n");
    }

    template <class ExecutionPolicy, typename DofViewA, typename DofViewB>
    T dotProduct(ExecutionPolicy&& policy, DofViewA a, DofViewB b) {
      constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
      using ValueT = typename std::iterator_traits<RM_CVREF_T(std::begin(a))>::value_type;
      auto dofSqr = dof_view<space, dim>(dofSqr_);
      DofCompwiseOp{multiplies<void>{}}(policy, a, b, dofSqr);
      reduce(policy, std::begin(dofSqr), std::end(dofSqr),
             std::begin(dof_view<space, dim>(normSqr_)), 0, plus<ValueT>{});
      return normSqr_.clone({memsrc_e::host, -1})[0];
    }

    template <class ExecutionPolicy, typename M, typename XView, typename BView>
    int solve(ExecutionPolicy&& policy, M&& A, XView&& xinout, BView&& b) {
      constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
      resize(xinout.numEntries());

      auto x = dof_view<space, dim>(x_);
      policy(range(numDofs), DofAssign{xinout, x});

      auto rr = dof_view<space, dim>(or_), r = dof_view<space, dim>(r_),
           p = dof_view<space, dim>(p_), Ap = dof_view<space, dim>(Ap_),
           Ar = dof_view<space, dim>(Ar_), MAp = dof_view<space, dim>(MAp_);
      T rArprev = 0, rAr, ApMAp;

      int iter = 0;
      auto shouldPrint = [](bool v = false) { return v; };
      T alpha, beta, cn0, cn, residualPreconditionedNorm;

      A.multiply(policy, x, Ap);
      DofCompwiseOp{minus<void>{}}(policy, b, Ap, rr);
      if (shouldPrint()) {
        auto res = dotProduct(policy, rr, rr);
        fmt::print("(after minus Ax) normSqr rhs: {}\n", res);
      }
      fmt::print("check num dofs: {}, r dofs: {}\n", x.numEntries(), r.numEntries());
      A.project(policy, rr);
      if (shouldPrint()) {
        auto res = dotProduct(policy, r, r);
        fmt::print("(after proj) normSqr rhs: {}\n", res);
      }

      A.precondition(policy, rr, r);
      if (shouldPrint()) {
        auto res = dotProduct(policy, r, r);
        fmt::print("(after precondition) normSqr rhs: {}\n", res);
      }
      fmt::print("done precondition, r dofs: {}\n", r.numEntries());
      policy(range(numDofs), DofAssign{r, p});

      A.multiply(policy, r, Ap);

      cn0 = dotProduct(policy, b, b);
      tol = tol * cn0 < tol ? tol * cn0 : tol;
      for (; iter != maxIters; ++iter) {
        cn = dotProduct(policy, rr, rr);
        if (shouldPrint(iter % 10 == 9)) {
          fmt::print("iter: {}, norm: {}, tol: {}\n", iter, cn, tol);
          getchar();
        }
        if (cn <= tol) {
          // print(xinout);
          break;
        }
        A.multiply(policy, r, Ar);
        if (shouldPrint(iter % 10 == 9)) fmt::print("iter: {}, done multiply\n", iter);
        A.project(policy, Ar);
        if (shouldPrint(iter % 10 == 9)) fmt::print("iter: {}, done project\n", iter);

        rAr = dotProduct(policy, r, Ar);
        if (shouldPrint(iter % 10 == 9)) fmt::print("iter: {}, alpha {}\n", iter, rAr);
        if (rAr == 0) break;

        if (rArprev != 0) {
          beta = rAr / rArprev;
          DofCompwiseOp{LinearCombineOp(beta)}(policy, p, r, p);     // p = r + beta * p;
          DofCompwiseOp{LinearCombineOp(beta)}(policy, Ap, Ar, Ap);  // Ap = Ar + beta * Ap;
        }

        A.precondition(policy, Ap, MAp);
        ApMAp = dotProduct(policy, MAp, Ap);
        if (ApMAp == 0) break;

        alpha = rAr / ApMAp;
        DofCompwiseOp{LinearCombineOp(alpha)}(policy, p, x, x);
        DofCompwiseOp{LinearCombineOp(-alpha)}(policy, Ap, rr, rr);
        DofCompwiseOp{LinearCombineOp(-alpha)}(policy, MAp, r, r);

        rArprev = rAr;
      }
      policy(range(numDofs), DofAssign{x, xinout});
      if (iter > 10) getchar();
      return iter;
    }
  };

}  // namespace zs