#pragma once
#include "LinearOperators.hpp"
#include "zensim/resource/Resource.h"
#include <cmath>

namespace zs {

	/// Bow/Math/LinearSolver/ConjugateGradient.h
  template <typename T, int dim, typename Index = std::size_t> struct ConjugateGradient {
    using TV = Vector<T, Index>;
    using allocator_type = ZSPmrAllocator<>;
    using size_type = std::make_unsigned_t<Index>;

    int maxIters;
    TV x_, r_, p_, q_, temp_;
    TV mr_, s_;
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
          numDofs{ndofs},
          tol{is_same_v<T, float> ? (T)1e-6 : (T)1e-12},
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
    }

    template <typename DV> void print(DV&& dv) {
      for (std::size_t i = 0; i != dv.size(); ++i) fmt::print("{} ", dv.get(i));
      fmt::print("\n");
    }

    template <class ExecutionPolicy, typename M, typename X, typename B>
    int solve(ExecutionPolicy&& policy, M&& A, X&& x0_, const B& b_) {
      constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
      auto xinout = dof_view<space>(x0_);
      resize(xinout.numEntries());
      auto x = dof_view<space>(x_);
      policy(range(numDofs), DofAssign{xinout, x});

      auto b = dof_view<space>(b_);
      auto r = dof_view<space>(r_), p = dof_view<space>(p_), q = dof_view<space>(q_),
           temp = dof_view<space>(temp_);
      auto mr = dof_view<space>(mr_), s = dof_view<space>(s_);

      int iter = 0;
      T alpha, beta, residualPreconditionedNorm, zTrk, zTrkLast;

      A.multiply(policy, x, temp);
      DofCompwiseOp{std::minus<void>{}}(policy, b, temp, r);  // r = b - temp;
      A.project(policy, r);
      A.precondition(policy, r, q);  // NOTE: requires that preconditioning matrix is projected
      policy(range(numDofs), DofAssign{q, p});  // p = q;

      zTrk = std::abs(DofDot{}(policy, r_, q_));  // zTrk = std::abs(dotProduct(r, q));
      residualPreconditionedNorm = std::sqrt(zTrk);
      T localTol = std::min(relTol * residualPreconditionedNorm, tol);
      for (; iter != maxIters; ++iter) {
        if (residualPreconditionedNorm <= localTol) {
          policy(range(numDofs), DofAssign{x, xinout});
          ///
          print(xinout);
          ///
          return iter;
        }
        A.multiply(policy, p, temp);
        A.project(policy, temp);
        alpha = zTrk / DofDot{}(policy, temp_, p_);  // alpha = zTrk / dotProduct(temp, p);

        DofCompwiseOp{LinearCombineOp(alpha)}(policy, p, x, x);      // x = x + alpha * p;
        DofCompwiseOp{LinearCombineOp(-alpha)}(policy, temp, r, r);  // r = r - alpha * temp;
        A.precondition(policy, r, q);  // NOTE: requires that preconditioning matrix is projected

        zTrkLast = zTrk;
        zTrk = DofDot{}(policy, q_, r_);  // zTrk = dotProduct(q, r);
        beta = zTrk / zTrkLast;

        DofCompwiseOp{LinearCombineOp(beta)}(policy, p, q, p);  // p = q + beta * p;

        residualPreconditionedNorm = std::sqrt(zTrk);
      }
    }
  };

}