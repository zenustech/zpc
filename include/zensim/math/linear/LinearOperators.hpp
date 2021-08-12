#pragma once
#include "zensim/types/View.h"
#include "zensim/types/Polymorphism.h"
#include "zensim/container/Vector.hpp"
#include "zensim/meta/Functional.h"

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

  template <typename T> struct LinearCombineOp {
    constexpr LinearCombineOp(T m = (T)1, T n = (T)1) noexcept : m{m}, n{n} {}
    template <typename L, typename R> constexpr auto operator()(L&& lhs, R&& rhs)
        -> decltype(std::declval<T>() * FWD(lhs) + std::declval<T>() * FWD(rhs)) {
      return m * FWD(lhs) + n * FWD(rhs);
    }
    T m, n;
  };

  struct DofCompwiseOp {
    using Ops = variant<std::plus<void>, std::multiplies<void>, std::minus<void>,
                        std::divides<void>, LinearCombineOp<float>, LinearCombineOp<double>>;
    template <typename Op> DofCompwiseOp(Op op) : _op{op} {}

    template <typename DofViewA, typename DofViewB, typename DofViewC, typename Op>
    struct ComputeOp {
      using Index = std::common_type_t<typename DofViewA::size_type, typename DofViewB::size_type,
                                       typename DofViewC::size_type>;
      ComputeOp(DofViewA a, DofViewB b, DofViewC c, Op op) : va{a}, vb{b}, vc{c}, op{op} {}
      constexpr void operator()(Index i) {
        vc.set(i, op(va.get(i, scalar_v), vb.get(i, scalar_v)));
      }

      DofViewA va;
      DofViewB vb;
      DofViewC vc;
      Op op;
    };

    template <class ExecutionPolicy, typename DofViewA, typename DofViewB, typename DofViewC>
    void operator()(ExecutionPolicy&& policy, DofViewA va, DofViewB vb, DofViewC vc) {
      match([&](auto op) {
        constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;

        if (va.numEntries() != vb.numEntries() || va.numEntries() != vc.numEntries())
          throw std::runtime_error("dof mismatch!");
        policy(range(va.numEntries()), ComputeOp{va, vb, vc, op});
      })(_op);
    }

    Ops _op;
  };
  struct DofCompwiseUnaryOp {
    using Ops = variant<std::negate<void>>;
    template <typename Op> DofCompwiseUnaryOp(Op op) : _op{op} {}

    template <typename DofViewA, typename DofViewB, typename Op> struct ComputeOp {
      using Index = std::common_type_t<typename DofViewA::size_type, typename DofViewB::size_type>;
      ComputeOp(DofViewA a, DofViewB b, Op op) : va{a}, vb{b}, op{op} {}
      constexpr void operator()(Index i) { vb.set(i, op(va.get(i, scalar_v))); }

      DofViewA va;
      DofViewB vb;
      Op op;
    };

    template <class ExecutionPolicy, typename DofViewA, typename DofViewB>
    void operator()(ExecutionPolicy&& policy, DofViewA va, DofViewB vb) {
      match([&](auto op) {
        constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
        if (va.numEntries() != vb.numEntries()) throw std::runtime_error("dof mismatch!");
        policy(range(va.numEntries()), ComputeOp{va, vb, op});
      })(_op);
    }

    Ops _op;
  };

  struct DofDot {
    template <class ExecutionPolicy, typename DofA, typename DofB>
    auto operator()(ExecutionPolicy&& policy, const DofA& a, const DofB& b) {
      constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
      auto va = dof_view<space>(a);
      auto vb = dof_view<space>(b);
      if (va.numEntries() != vb.numEntries()) throw std::runtime_error("dof mismatch!");

      auto tmp = a;
      auto vtmp = dof_view<space>(tmp);
      using ValueT = typename std::iterator_traits<RM_CVREF_T(std::begin(vtmp))>::value_type;

      DofCompwiseOp transOp{std::multiplies<void>{}};
      transOp(policy, va, vb, vtmp);

      Vector<ValueT> init{1, memsrc_e::host};
      init[0] = (ValueT)0;
      Vector<ValueT> res = init.clone(b.allocator());
      reduce(policy, std::begin(vtmp), std::end(vtmp), std::begin(res), (ValueT)0,
             std::plus<ValueT>{});
      return res.clone({memsrc_e::host, -1})[0];
    }
    template <class ExecutionPolicy, typename Dof>
    auto operator()(ExecutionPolicy&& policy, const Dof& a) {
      constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
      using ValueA =
          typename std::iterator_traits<RM_CVREF_T(std::begin(dof_view<space>(a)))>::value_type;

      auto va = dof_view<space>(a);
      auto tmp = a;
      auto vtmp = dof_view<space>(tmp);

      DofCompwiseOp transOp{std::multiplies<void>{}};
      transOp(policy, va, va, vtmp);

      Vector<ValueA> init{1, memsrc_e::host};
      init[0] = 0;
      Vector<ValueA> res = init.clone(a.allocator());
      // fmt::print("soon to be reducing {} entries\n", va.numEntries());
      reduce(policy, std::begin(vtmp), std::end(vtmp), std::begin(res), (ValueA)0,
             std::plus<ValueA>{});
      return res.clone({memsrc_e::host, -1})[0];
    }
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