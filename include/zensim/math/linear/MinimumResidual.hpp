#pragma once
#include <cmath>

#include "LinearOperators.hpp"
#include "zensim/math/matrix/Givens.hpp"

namespace zs {

  /// Bow/Math/LinearSolver/ConjugateGradient.h
  template <typename T, int dim, typename Index = zs::size_t> struct MinRes {
    using TV = Vector<T>;
    using allocator_type = ZSPmrAllocator<>;
    using size_type = zs::make_unsigned_t<Index>;

    math::GivensRotation<T> Gk, Gkm1, Gkm2;
    T gamma, delta, epsilon;
    T beta_kp1, alpha_k, beta_k, sk;
    TV x_, mk_, mkm1_, mkm2_;
    TV z_, qkp1_, qk_, qkm1_;
    T tk;

    using TV2 = vec<T, 2>;
    TV2 last_two_components_of_givens_transformed_least_squares_rhs;
    T rhsNorm2;

    int maxIters;
    // for dot
    TV dofSqr_;
    TV normSqr_;
    size_type numDofs;
    T tol;
    T relTol;

    MinRes(const allocator_type& allocator, size_type ndofs)
        : x_{allocator, ndofs},
          mk_{allocator, ndofs},
          mkm1_{allocator, ndofs},
          mkm2_{allocator, ndofs},
          z_{allocator, ndofs},
          qkp1_{allocator, ndofs},
          qk_{allocator, ndofs},
          qkm1_{allocator, ndofs},
          dofSqr_{allocator, ndofs},
          normSqr_{allocator, 1},
          numDofs{ndofs},
          tol{is_same_v<T, float> ? (T)1e-6 : (T)1e-12},
          maxIters{1000},
          relTol{0.5f} {}
    MinRes(memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : MinRes{get_memory_source(mre, devid), (size_type)0} {}
    MinRes(size_type count, memsrc_e mre = memsrc_e::host, ProcID devid = -1)
        : MinRes{get_memory_source(mre, devid), count} {}

    void resize(size_type ndofs) {
      numDofs = ndofs;
      x_.resize(ndofs);
      mk_.resize(ndofs);
      mkm1_.resize(ndofs);
      mkm2_.resize(ndofs);
      z_.resize(ndofs);
      qkp1_.resize(ndofs);
      qk_.resize(ndofs);
      qkm1_.resize(ndofs);
      dofSqr_.resize(ndofs);
      // set zeros?
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

    T applyAllPreviousGivensRotationsAndDetermineNewGivens() {
      // QR the LHS: gamma, delta, epsilon
      Gkm2 = Gkm1;
      Gkm1 = Gk;
      TV2 epsilon_k_and_phi_k{0, beta_k};
      Gkm2.vecRotation(epsilon_k_and_phi_k);

      epsilon = epsilon_k_and_phi_k(0);
      TV2 delta_k_and_zsi_k{epsilon_k_and_phi_k(1), alpha_k};
      Gkm1.vecRotation(delta_k_and_zsi_k);
      delta = delta_k_and_zsi_k(0);
      TV2 temp{delta_k_and_zsi_k(1), beta_kp1};

      Gk.compute(temp(0), temp(1));
      Gk.vecRotation(temp);
      gamma = temp(0);

      // Now deal with the RHS: tk and residual (two norm)
      Gk.vecRotation(last_two_components_of_givens_transformed_least_squares_rhs);
      tk = last_two_components_of_givens_transformed_least_squares_rhs(0);
      T residual = last_two_components_of_givens_transformed_least_squares_rhs(
          1);  // This is the two norm of the residual.
      last_two_components_of_givens_transformed_least_squares_rhs
          = TV2{residual, (T)0};  // Set up for the next iteration
      if (residual < 0)
        return -residual;
      else
        return residual;
    }

    template <class ExecutionPolicy, typename M, typename XView, typename BView>
    int solve(ExecutionPolicy&& policy, M&& A, XView&& xinout, BView&& b) {
      constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
      resize(xinout.numEntries());

      auto x = dof_view<space, dim>(x_);
      policy(range(numDofs), DofAssign{xinout, x});

      auto mk = dof_view<space, dim>(mk_), mkm1 = dof_view<space, dim>(mkm1_),
           mkm2 = dof_view<space, dim>(mkm2_), z = dof_view<space, dim>(z_),
           qkp1 = dof_view<space, dim>(qkp1_), qk = dof_view<space, dim>(qk_),
           qkm1 = dof_view<space, dim>(qkm1_);

      int iter = 0;

      A.multiply(policy, x, qkp1);
      DofCompwiseOp{minus<void>{}}(policy, b, qkp1, qkp1);

      A.project(policy, qkp1);
      A.precondition(policy, qkp1, z);

      T residualPreconditionedNorm = std::sqrt(dotProduct(policy, z, qkp1));
      beta_kp1 = residualPreconditionedNorm;

      T localTol = std::min(relTol * residualPreconditionedNorm, tol);

      if (residualPreconditionedNorm < localTol || residualPreconditionedNorm == 0) {
        policy(range(numDofs), DofAssign{x, xinout});
        return iter;
      }
      if (residualPreconditionedNorm > 0) {
        DofCompwiseCustomUnaryOp{divides<void>{}, beta_kp1}(policy, qkp1,
                                                                 qkp1);  // qkp1 /= beta_kp1;
        DofCompwiseCustomUnaryOp{divides<void>{}, beta_kp1}(policy, z, z);  // z /= beta_kp1;
      }
      last_two_components_of_givens_transformed_least_squares_rhs
          = TV2{residualPreconditionedNorm, 0};
      for (; iter != maxIters; ++iter) {
        // fmt::print("iter {}, residualNorm: {} ({})\n", iter, residualPreconditionedNorm,
        // localTol);
        if (residualPreconditionedNorm < localTol) break;

        // use mk to store zk to save storage
        mkm2.swap(mkm1);
        mkm1.swap(mk);
        policy(range(numDofs), DofAssign{z, mk});  // mk = z;

        beta_k = beta_kp1;

        qkm1.swap(qkp1);
        qkm1.swap(qk);

        A.multiply(policy, mk, qkp1);
        A.project(policy, qkp1);
        alpha_k = dotProduct(policy, mk, qkp1);

        DofCompwiseOp{LinearCombineOp(-alpha_k)}(policy, qk, qkp1, qkp1);
        DofCompwiseOp{LinearCombineOp(-beta_k)}(policy, qkm1, qkp1, qkp1);

        A.precondition(policy, qkp1, z);
        beta_kp1 = std::sqrt(std::max((T)0, dotProduct(policy, z, qkp1)));

        if (beta_kp1 > 0) {
          DofCompwiseCustomUnaryOp{divides<void>{}, beta_kp1}(policy, qkp1,
                                                                   qkp1);  // qkp1 /= beta_kp1;
          DofCompwiseCustomUnaryOp{divides<void>{}, beta_kp1}(policy, z, z);  // z /= beta_kp1;
        }
        residualPreconditionedNorm = applyAllPreviousGivensRotationsAndDetermineNewGivens();

        DofCompwiseOp{LinearCombineOp((T)1, -delta)}(policy, mk, mkm1, mk);
        DofCompwiseOp{LinearCombineOp((T)1, -epsilon)}(policy, mk, mkm2, mk);
        DofCompwiseCustomUnaryOp{divides<void>{}, gamma}(policy, mk, mk);  // mk /= gamma

        DofCompwiseOp{LinearCombineOp(tk)}(policy, mk, x, x);
      }
      policy(range(numDofs), DofAssign{x, xinout});
      if (iter > 10) getchar();
      return iter;
    }
  };

}  // namespace zs