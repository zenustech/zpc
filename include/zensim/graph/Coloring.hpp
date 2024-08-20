#pragma once
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/math/matrix/SparseMatrix.hpp"

namespace zs {

  /// @note assume the graph is undirected
  template <typename Policy, typename T, bool RowMajor, typename Ti, typename Tn,
            typename AllocatorT, typename WeightRangeT, typename ColorRangeT>
  inline Ti fast_independent_sets(Policy &&policy,
                                  const SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT> &spmat,
                                  WeightRangeT &&weights, ColorRangeT &&colors) {
    using ValT = RM_CVREF_T(*std::begin(weights));
    static_assert(std::is_arithmetic_v<ValT>, "weight type should be arithmetic");

    using ColorT = RM_CVREF_T(*std::begin(colors));
    constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
    static_assert(std::is_arithmetic_v<ColorT>, "color type should be arithmetic");

    auto n = range_size(weights);
    if (n != spmat.rows() || n != spmat.cols())
      throw std::runtime_error("spmat and weight size mismatch");
    if (!valid_memspace_for_execution(policy, spmat.get_allocator()))
      throw std::runtime_error("current memory location not compatible with the execution policy");
    /// @note compilation checks
    // assert_backend_presence<space>();

    bool shouldSync = policy.shouldSync();
    policy.sync(true);

    policy(colors, [] ZS_LAMBDA(ColorT & color) { color = detail::deduce_numeric_max<ColorT>(); });

    auto allocator = get_temporary_memory_source(policy);
    zs::Vector<int> done{allocator, 2};
    zs::Vector<u8> maskOut{allocator, (size_t)n};
    maskOut.reset(0);
    // policy(maskOut, [] ZS_LAMBDA(u8 & mask) { mask = 0; });
    std::vector<int> hdone(2);
    ColorT color;
    for (color = 0;; color += 2) {
      done.reset(0);
      // policy(done, [] ZS_LAMBDA(int &v) { v = 0; });
      policy(range(n), [spmat = proxy<space>(spmat), ws = std::begin(weights),
                        colors = std::begin(colors), done = view<space>(done),
                        maskOut = view<space>(maskOut), color] ZS_LAMBDA(Ti row) mutable {
        // if already colored, exit
        if (maskOut[row]) return;

        auto w = ws[row];
        auto bg = spmat._ptrs[row];
        auto ed = spmat._ptrs[row + 1];
        bool colorMax = true;
        bool colorMin = true;
        for (auto k = bg; k != ed; ++k) {
          auto neighbor = spmat._inds[k];
          // skip the already colored neighbor nodes
          if (!maskOut[neighbor]) {
            auto ow = ws[neighbor];
            if (ow > w) colorMax = false;
            if (ow < w) colorMin = false;
          }
        }
        if (colorMax) {
          colors[row] = color + 1;
          done[0] = 1;
        } else if (colorMin) {
          colors[row] = color + 2;
          done[1] = 1;
        }
      });
      /// @note policy is executing in synchronous fashion, safe to retrieve value here
      done.retrieveVals(hdone.data());
      if (hdone[0] == 0) {
        break;
      } else if (hdone[1] == 0) {
        color++;
        break;
      }
      policy(range(n), [colors = std::begin(colors), maskOut = view<space>(maskOut),
                        color] ZS_LAMBDA(Ti row) mutable {
        if (maskOut[row]) return;
        if (colors[row] == color + 1 || colors[row] == color + 2) maskOut[row] = 1;
      });
    }

    policy.sync(shouldSync);
    return color;
  }

  /// @note assume the graph is undirected
  /// @note not necessarily produces less colors than the above one
  template <typename Policy, typename T, bool RowMajor, typename Ti, typename Tn,
            typename AllocatorT, typename WeightRangeT, typename ColorRangeT>
  inline Ti maximum_independent_sets(Policy &&policy,
                                     const SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT> &spmat,
                                     WeightRangeT &&weights, ColorRangeT &&colors) {
    using ValT = RM_CVREF_T(*std::begin(weights));
    static_assert(std::is_arithmetic_v<ValT>, "weight type should be arithmetic");

    using ColorT = RM_CVREF_T(*std::begin(colors));
    constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
    static_assert(std::is_arithmetic_v<ColorT>, "color type should be arithmetic");

    auto n = range_size(weights);
    if (n != spmat.rows() || n != spmat.cols())
      throw std::runtime_error("spmat and weight size mismatch");
    if (!valid_memspace_for_execution(policy, spmat.get_allocator()))
      throw std::runtime_error("current memory location not compatible with the execution policy");
    /// @note compilation checks
    // assert_backend_presence<space>();

    bool shouldSync = policy.shouldSync();
    policy.sync(true);

    policy(colors, [] ZS_LAMBDA(ColorT & color) { color = detail::deduce_numeric_max<ColorT>(); });

    // @note 0: free, 1: colored, 2: temporaral exclusion (reset upon the next color iteration)
    auto allocator = get_temporary_memory_source(policy);
    zs::Vector<u8> maskOut{allocator, (size_t)n};
    maskOut.reset(0);

    // @note coloring occured (including maximum set expansion)
    // @note 0, 1, 2
    zs::Vector<int> expanded{allocator, 1};

    ColorT color;
    for (color = 0;; color++) {
      expanded.reset(0);
      policy(range(n), [spmat = proxy<space>(spmat), ws = std::begin(weights),
                        colors = std::begin(colors), expanded = view<space>(expanded),
                        maskOut = view<space>(maskOut), color] ZS_LAMBDA(Ti row) mutable {
        /// if already colored, exit
        if (maskOut[row]) return;

        auto w = ws[row];
        auto bg = spmat._ptrs[row];
        auto ed = spmat._ptrs[row + 1];
        bool colorMin = true;
        for (auto k = bg; k != ed; ++k) {
          auto neighbor = spmat._inds[k];
          // skip the already colored neighbor nodes
          if (!maskOut[neighbor]) {
            if (ws[neighbor] < w) {
              colorMin = false;
              break;
            }
          }
        }
        if (colorMin) {
          colors[row] = color + 1;
          expanded[0] = 1;
        }
      });
      if (expanded.getVal() == 0) break;

      policy(range(n), [colors = std::begin(colors), maskOut = view<space>(maskOut),
                        color] ZS_LAMBDA(Ti row) mutable {
        if (maskOut[row]) return;
        if (colors[row] == color + 1) maskOut[row] = 1;
      });

      /// iterative expansion
      do {
        expanded.reset(0);
        // mark nodes that are neighbors of nodes K with mask 1 (maskOut[k] == 1)
        policy(range(n), [spmat = proxy<space>(spmat), colors = std::begin(colors),
                          maskOut = view<space>(maskOut), color] ZS_LAMBDA(Ti row) mutable {
          if (maskOut[row]) return;
          auto bg = spmat._ptrs[row];
          auto ed = spmat._ptrs[row + 1];
          for (auto k = bg; k != ed; ++k)
            if (colors[spmat._inds[k]] == color + 1) {
              maskOut[row] = 2;
              return;
            }
        });
        // continue the expansion of the current set
        policy(range(n), [spmat = proxy<space>(spmat), ws = std::begin(weights),
                          colors = std::begin(colors), expanded = view<space>(expanded),
                          maskOut = view<space>(maskOut), color] ZS_LAMBDA(Ti row) mutable {
          // if already colored or excluded, exit
          if (maskOut[row]) return;

          auto w = ws[row];
          auto bg = spmat._ptrs[row];
          auto ed = spmat._ptrs[row + 1];
          bool colorMin = true;
          for (auto k = bg; k != ed; ++k) {
            auto neighbor = spmat._inds[k];
            if (!maskOut[neighbor]) {
              if (ws[neighbor] < w) {
                colorMin = false;
                break;
              }
            }
          }
          if (colorMin) {
            colors[row] = color + 1;
            expanded[0] = 1;
          }
        });

        policy(range(n), [colors = std::begin(colors), maskOut = view<space>(maskOut),
                          color] ZS_LAMBDA(Ti row) mutable {
          if (maskOut[row]) return;
          if (colors[row] == color + 1) maskOut[row] = 1;
        });

      } while (expanded.getVal() == 1);

      /// reset maskOut with 2
      policy(maskOut, [] ZS_LAMBDA(u8 & mask) {
        if (mask == 2) mask = 0;
      });
    }

    policy.sync(shouldSync);
    return color;
  }

}  // namespace zs