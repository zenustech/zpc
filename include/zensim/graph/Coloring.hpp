#pragma once
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/math/matrix/SparseMatrix.hpp"

namespace zs {

  /// @note assume the graph is undirected
  template <typename Policy, typename T, bool RowMajor, typename Ti, typename Tn,
            typename AllocatorT, typename WeightRangeT, typename ColorRangeT>
  inline Ti fast_independent_set(Policy &&policy,
                                 const SparseMatrix<T, RowMajor, Ti, Tn, AllocatorT> &spmat,
                                 WeightRangeT &&weights, ColorRangeT &&colors) {
    using ValT = RM_CVREF_T(*std::begin(weights));
    static_assert(std::is_arithmetic_v<ValT>, "weight type should be arithmetic");

    using ColorT = RM_CVREF_T(*std::begin(colors));
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    static_assert(std::is_arithmetic_v<ColorT>, "color type should be arithmetic");

    auto n = range_size(weights);
    if (n != spmat.rows() || n != spmat.cols())
      throw std::runtime_error("spmat and weight size mismatch");
    if (!valid_memspace_for_execution(policy, spmat.get_allocator()))
      throw std::runtime_error("current memory location not compatible with the execution policy");
    /// @note compilation checks
    // assert_backend_presence<space>();

    policy(colors, [] ZS_LAMBDA(ColorT & color) { color = limits<ColorT>::max(); });

    zs::Vector<int, AllocatorT> done{spmat.get_allocator(), 2};
    std::vector<int> hdone(2);
    ColorT color;
    for (color = 0;; color += 2) {
      done.reset(0);
      policy(range(n),
             [spmat = proxy<space>(spmat), ws = std::begin(weights), colors = std::begin(colors),
              done = view<space>(done), color] ZS_LAMBDA(Ti row) mutable {
               // if already colored, exit
               if (colors[row] != limits<ColorT>::max()) return;

               auto w = ws[row];
               auto bg = spmat._ptrs[row];
               auto ed = spmat._ptrs[row + 1];
               bool colorMax = true;
               bool colorMin = true;
               for (auto k = bg; k != ed; ++k) {
                 auto neighbor = spmat._inds[k];
                 // skip the already colored neighbor nodes
                 if (colors[neighbor] == limits<ColorT>::max()) {
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
      done.retrieveVals(hdone.data());
      if (hdone[0] == 0) {
        break;
      } else if (hdone[1] == 0) {
        color++;
        break;
      }
    }
    return color;
  }

}  // namespace zs