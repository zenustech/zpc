#pragma once
#include "initialization.hpp"
#include "zensim/ZpcBuiltin.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/zpc_tpls/fmt/core.h"

namespace zs {

  template <typename Pol, typename Range, auto space = remove_reference_t<Pol>::exec_tag::value>
  bool test_binary_search(Pol &&policy, Range &&r) {
    static_assert(is_lvalue_reference_v<decltype(*std::begin(r))>,
                  "deref of the iterator shall be a lvalue");
    using value_t = RM_REF_T(*std::begin(r));
    static_assert(is_fundamental_v<value_t>, "value_t should be a fundamental type");
    const auto sz = range_size(r);

    if (sz == 0) return true;

    auto allocator = get_temporary_memory_source(policy);
    /// generate search inputs
    Vector<value_t> samples{allocator, (size_t)(sz + 2)};
    policy(enumerate(r),
           [sz, samples = view<space>(samples)] ZS_LAMBDA(size_t i, const value_t &v) mutable {
             samples[i + 1] = v;
             if (i == 0) samples[0] = v - 1;
             if (i == sz - 1) samples[sz + 1] = v + 1;
           });
    /// search results
    Vector<int> res{allocator, (size_t)(sz + 2)}, groundTruth{(size_t)(sz + 2)};

// binary search
#if 0
    policy(enumerate(samples), [r, sz, res = view<space>(res)](int i, const auto &v) ZS_LAMBDA {
      using index_type = int;
      index_type left = 0, right = sz;
      while (left < right) {
        auto mid = left + (right - left) / 2;
        if (r[mid] > v)
          right = mid;
        else
          left = mid + 1;
      }
      if (left < sz) {
        if (r[left] > v) left--;
      } else
        left = sz - 1;
      // left could be -1
      res[i] = left;
    });
#elif 1
    policy(enumerate(samples), [r, sz, res = view<space>(res)](int i, const auto &v) ZS_LAMBDA {
      using index_type = int;
      index_type left = 0, right = sz;  // [left, right)
      while (left <= right) {
        auto mid = left + (right - left) / 2;
        if (r[mid] > v)
          right = mid;
        else
          left = mid + 1;
      }
      res[i] = right - 1;
    });
#else
    policy(enumerate(samples), [r, sz, res = view<space>(res)](int i, const auto &v) ZS_LAMBDA {
      using index_type = int;
      index_type left = 0, right = sz - 1;  // [left, right]
      while (left <= right) {
        auto mid = left + (right - left) / 2;
        if (r[mid] > v)
          right = mid - 1;
        else
          left = mid + 1;
      }
      res[i] = right;
    });
#endif
    res = res.clone({memsrc_e::host, -1});

    // ground truth
    samples = samples.clone({memsrc_e::host, -1});
    Vector<value_t> vals{allocator, (size_t)sz};
    policy(zip(r, vals), [] ZS_LAMBDA(const value_t &src, value_t &dst) mutable { dst = src; });
    vals = vals.clone({memsrc_e::host, -1});

    auto hostPol = preferred_host_policy();
    hostPol(enumerate(samples), [&](int i, const auto &sample) {
      // if (sample >= vals[sz - 1]) groundTruth[i] = sz - 1;
      if (sample < vals[0])
        groundTruth[i] = -1;
      else {
        int j = 1;
        for (; j < sz; ++j)
          if (sample < vals[j]) break;
        groundTruth[i] = j - 1;
      }
    });

    if (sz < 20) {
      fmt::print("sequence: ");
      for (int i = 0; i < sz; ++i) fmt::print("{} ", vals[i]);
      fmt::print("\n");
    }

    bool valid = true;
    hostPol(enumerate(res, groundTruth), [&](int i, int loc, int ref) {
      if (loc != ref) valid = false;
      if (sz < 20) {
        fmt::print("{}-th sample [{}], result: {} (ref: {})\n", i, samples[i], loc, ref);
      }
    });

    return valid;
  }

}  // namespace zs