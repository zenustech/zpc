#pragma once
#include "zensim/ZpcBuiltin.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  template <typename Pol, typename Range, typename Op,
            auto space = remove_reference_t<Pol>::exec_tag::value>
  bool test_reduction(Pol &&policy, Range &&r, Op op) {
    static_assert(is_lvalue_reference_v<decltype(*std::begin(r))>,
                  "deref of the iterator shall be a lvalue");
    using value_t = RM_REF_T(*std::begin(r));
    static_assert(is_fundamental_v<value_t>, "value_t should be a fundamental type");
    const auto sz = range_size(r);

    auto mop = make_monoid(op);
    auto allocator = get_temporary_memory_source(policy);
    Vector<value_t> res{allocator, 1};
    reduce(policy, std::begin(r), std::end(r), std::begin(res), mop.identity(), op);

    Vector<value_t> vals{allocator, (size_t)sz};
    policy(zip(r, vals), [] ZS_LAMBDA(const value_t &src, value_t &dst) mutable { dst = src; });
    vals = vals.clone({memsrc_e::host, -1});
    value_t e = mop.identity();
    for (size_t i = 0; i != sz; ++i) e = op(e, vals[i]);

    if constexpr (is_integral_v<value_t>) {
      return e == res.getVal();
    } else {
      return std::abs(e - res.getVal()) / e < 1e-6;
    }
    return 0;
  }

}  // namespace zs