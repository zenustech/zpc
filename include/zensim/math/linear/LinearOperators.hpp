#pragma once
#include "zensim/container/Vector.hpp"
#include "zensim/meta/Functional.h"
#include "zensim/types/Polymorphism.h"
#include "zensim/types/View.h"

namespace zs {

  /*
    unlike proxy<space>(ranges)...
    dof_view can usually be constructed in multiple ways
    thus we leave that burden outside the operator ctors
  */
  template <typename DofViewA, typename DofViewB> struct DofAssign {
    constexpr DofAssign(DofViewA a, DofViewB b) : dofa{a}, dofb{b} {}
    using size_type = typename DofViewB::size_type;
    constexpr void operator()(size_type i) { dofb.set(i, dofa.get(i, wrapv<DofViewB::entry_e>{})); }

    DofViewA dofa;
    DofViewB dofb;
  };
  template <typename DofView, typename V> struct DofFill {
    using size_type = typename DofView::size_type;
    using value_type = V;
    constexpr DofFill(DofView a, V v) : dof{a}, v{v} {}
    constexpr void operator()(size_type i) { dof.set(i, v); }

    DofView dof;
    V v;
  };

  template <typename T> struct LinearCombineOp {
    constexpr LinearCombineOp(T m = (T)1, T n = (T)1) noexcept : m{m}, n{n} {}
    template <typename L, typename R> constexpr auto operator()(L&& lhs, R&& rhs)
        -> decltype(declval<T>() * FWD(lhs) + declval<T>() * FWD(rhs)) {
      return m * FWD(lhs) + n * FWD(rhs);
    }
    T m, n;
  };

  struct DofCompwiseOp {
    using Ops = variant<plus<void>, multiplies<void>, minus<void>,
                        divides<void>, LinearCombineOp<float>, LinearCombineOp<double>>;
    template <typename Op> DofCompwiseOp(Op op) : _op{op} {}

    template <typename DofViewA, typename DofViewB, typename DofViewC, typename Op>
    struct ComputeOp {
      using Index = math::op_result_t<typename DofViewA::size_type, typename DofViewB::size_type,
                                      typename DofViewC::size_type>;
      ComputeOp(DofViewA a, DofViewB b, DofViewC c, Op op) : va{a}, vb{b}, vc{c}, op{op} {}
      constexpr void operator()(Index i) {
        vc.set(i, op(va.get(i, scalar_c), vb.get(i, scalar_c)));
      }

      DofViewA va;
      DofViewB vb;
      DofViewC vc;
      Op op;
    };

    template <class ExecutionPolicy, typename DofViewA, typename DofViewB, typename DofViewC>
    void operator()(ExecutionPolicy&& policy, DofViewA va, DofViewB vb, DofViewC vc) {
      match([&](auto op) {
        // constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
        if (va.numEntries() != vb.numEntries() || va.numEntries() != vc.numEntries())
          throw std::runtime_error("dof mismatch!");
        policy(range(va.numEntries()), ComputeOp{va, vb, vc, op});
      })(_op);
    }

    Ops _op;
  };
  template <template <typename> class Op, typename T> struct DofCompwiseCustomUnaryOp {
    constexpr DofCompwiseCustomUnaryOp(Op<void> op, T v) : op{op}, v{v} {}

    template <typename DofViewA, typename DofViewB> struct ComputeOp {
      using Index = std::common_type_t<typename DofViewA::size_type, typename DofViewB::size_type>;
      ComputeOp(DofViewA a, DofViewB b, Op<void> op, T v) : va{a}, vb{b}, op{op}, v{v} {}
      constexpr void operator()(Index i) { vb.set(i, op(va.get(i, scalar_c), v)); }

      DofViewA va;
      DofViewB vb;
      Op<void> op;
      T v;
    };

    template <class ExecutionPolicy, typename DofViewA, typename DofViewB>
    void operator()(ExecutionPolicy&& policy, DofViewA va, DofViewB vb) {
      constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
      if (va.numEntries() != vb.numEntries()) throw std::runtime_error("dof mismatch!");
      policy(range(va.numEntries()), ComputeOp<DofViewA, DofViewB>{va, vb, op, v});
    }

    Op<void> op;
    T v;
  };
  template <template <typename> class Op, typename T> DofCompwiseCustomUnaryOp(Op<void>, T)
      -> DofCompwiseCustomUnaryOp<Op, T>;

  struct DofCompwiseUnaryOp {
    using Ops = variant<std::negate<void>>;
    template <typename Op> DofCompwiseUnaryOp(Op op) : _op{op} {}

    template <typename DofViewA, typename DofViewB, typename Op> struct ComputeOp {
      using Index = std::common_type_t<typename DofViewA::size_type, typename DofViewB::size_type>;
      ComputeOp(DofViewA a, DofViewB b, Op op) : va{a}, vb{b}, op{op} {}
      constexpr void operator()(Index i) { vb.set(i, op(va.get(i, scalar_c))); }

      DofViewA va;
      DofViewB vb;
      Op op;
    };

    template <class ExecutionPolicy, typename DofViewA, typename DofViewB>
    void operator()(ExecutionPolicy&& policy, DofViewA va, DofViewB vb) {
      match([&](auto op) {
        constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
        if (va.numEntries() != vb.numEntries()) throw std::runtime_error("dof mismatch!");
        policy(range(va.numEntries()), ComputeOp{va, vb, op});
      })(_op);
    }

    Ops _op;
  };

  struct IdentitySystem {  ///< accepts dofviews as inputs
    template <class ExecutionPolicy, typename In, typename Out>
    void multiply(ExecutionPolicy&& policy, In&& in, Out&& out) {
      // DofCompwiseUnaryOp{std::negate<void>{}}(policy, FWD(in), FWD(out));
      policy(range(out.size()), DofAssign{FWD(in), FWD(out)});
    }

    template <class ExecutionPolicy, typename InOut>
    void project(ExecutionPolicy&& policy, InOut&& inout) {}

    template <class ExecutionPolicy, typename In, typename Out>
    void precondition(ExecutionPolicy&& policy, In&& in, Out&& out) {
      policy(range(out.size()), DofAssign{FWD(in), FWD(out)});
    }
  };

}  // namespace zs