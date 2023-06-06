#pragma once
#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  /// @note ref: https://userweb.cs.txstate.edu/~burtscher/research/ECL-CC/
  template <typename Policy, typename SpMatT, typename FaRange>
  void union_find(Policy&& pol, const SpMatT& spm, FaRange&& faRange) {
    using SpmvT = RM_CVREF_T(spm);
    using Ti = typename SpmvT::index_type;

    constexpr execspace_e space = RM_CVREF_T(pol)::exec_tag::value;
    auto n = spm.outerSize();
    /// @note init
    pol(range(n), [spmv = view<space>(spm), fas = std::begin(faRange)] ZS_LAMBDA(Ti v) mutable {
      Ti m = v;
      for (auto i = spmv._ptrs[v]; (m == v) && i < spmv._ptrs[v + 1]; ++i)
        m = m < spmv._inds[i] ? m : spmv._inds[i];
      fas[v] = m;
    });
    /// @note connect components
    pol(range(n), [spmv = view<space>(spm), fas = std::begin(faRange),
                   execTag = wrapv<space>{}] ZS_LAMBDA(Ti v) mutable {
      auto representative = [&nstat = fas](const Ti idx) -> Ti {
        Ti curr = nstat[idx];
        if (curr != idx) {
          Ti next, prev = idx;
          while (curr > (next = nstat[curr])) {
            nstat[prev] = next;
            prev = curr;
            curr = next;
          }
        }
        return curr;
      };
      Ti vstat = representative(v);
      for (int i = spmv._ptrs[v]; i < spmv._ptrs[v + 1]; i++) {
        const auto nli = spmv._inds[i];
        if (v > nli) {
          auto ostat = representative(nli);
          bool repeat;
          do {
            repeat = false;
            if (vstat != ostat) {
              Ti ret;
              if (vstat < ostat) {
                if ((ret = atomic_cas(execTag, &fas[ostat], ostat, vstat)) != ostat) {
                  ostat = ret;
                  repeat = true;
                }
              } else {
                if ((ret = atomic_cas(execTag, &fas[vstat], vstat, ostat)) != vstat) {
                  vstat = ret;
                  repeat = true;
                }
              }
            }
          } while (repeat);
        }
      }
    });
  }

  template <typename Policy, typename SpMatT, typename FaRange, typename Predicate>
  void union_find(Policy&& pol, const SpMatT& spm, FaRange&& faRange, Predicate&& skipPred) {
    using SpmvT = RM_CVREF_T(spm);
    using Ti = typename SpmvT::index_type;

    constexpr execspace_e space = RM_CVREF_T(pol)::exec_tag::value;
    auto n = spm.outerSize();
    /// @note init
    pol(range(n),
        [spmv = view<space>(spm), fas = std::begin(faRange), skipPred] ZS_LAMBDA(Ti v) mutable {
          Ti m = v;
          for (auto i = spmv._ptrs[v]; (m == v) && i < spmv._ptrs[v + 1]; ++i) {
            if (skipPred(spmv._vals[i])) continue;
            m = m < spmv._inds[i] ? m : spmv._inds[i];
          }
          fas[v] = m;
        });
    /// @note connect components
    pol(range(n), [spmv = view<space>(spm), fas = std::begin(faRange), execTag = wrapv<space>{},
                   skipPred] ZS_LAMBDA(Ti v) mutable {
      auto representative = [&nstat = fas](const Ti idx) -> Ti {
        Ti curr = nstat[idx];
        if (curr != idx) {
          Ti next, prev = idx;
          while (curr > (next = nstat[curr])) {
            nstat[prev] = next;
            prev = curr;
            curr = next;
          }
        }
        return curr;
      };
      Ti vstat = representative(v);
      for (int i = spmv._ptrs[v]; i < spmv._ptrs[v + 1]; i++) {
        if (skipPred(spmv._vals[i])) continue;
        const auto nli = spmv._inds[i];
        if (v > nli) {
          auto ostat = representative(nli);
          bool repeat;
          do {
            repeat = false;
            if (vstat != ostat) {
              Ti ret;
              if (vstat < ostat) {
                if ((ret = atomic_cas(execTag, &fas[ostat], ostat, vstat)) != ostat) {
                  ostat = ret;
                  repeat = true;
                }
              } else {
                if ((ret = atomic_cas(execTag, &fas[vstat], vstat, ostat)) != vstat) {
                  vstat = ret;
                  repeat = true;
                }
              }
            }
          } while (repeat);
        }
      }
    });
  }

}  // namespace zs