#pragma once
/// @credits Lu Shuliang
#include "zensim/container/Bht.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/math/matrix/SparseMatrix.hpp"

// #include <queue>
// #include <vector>

namespace zs {

  struct kernel_check_empty_frontier {
    template <typename ParamsT>
    constexpr void operator()(const int &isFrontier, ParamsT &params) noexcept {
      auto &[isFrontierEmpty] = params;
      if (isFrontier) isFrontierEmpty[0] = 0;
    }
  };

  struct kernel_bfs_augmented_path {
    template <typename ParamsT> constexpr void operator()(int vi, ParamsT &params) noexcept {
      auto &[visited, potentialFlows, sink, parents, frontier, isSinkFound, capacity, exec_tag]
          = params;
      if (!frontier[vi]) return;
      frontier[vi] = 0;
      for (int i = capacity._ptrs[vi]; i < capacity._ptrs[vi + 1]; ++i) {
        int nvi = capacity._inds[i];
        if (capacity._vals[i] <= 0) continue;
        if (atomic_cas(exec_tag, &visited[nvi], 0, 1) == 1) continue;

        frontier[nvi] = 1;
        parents[nvi] = vi;
        potentialFlows[nvi]
            = capacity._vals[i] < potentialFlows[vi] ? capacity._vals[i] : potentialFlows[vi];
        if (nvi == sink) isSinkFound[0] = 1;
      }
    }
  };

  template <typename Policy, typename SpMat, typename T = typename SpMat::value_type,
            typename Ti = typename SpMat::index_type>
  bool find_augmented_path(Policy &&pol, Ti source, Ti sink, const SpMat &capacity,
                           Vector<int> &parents, Vector<int> &visited, Vector<int> &frontier,
                           Vector<T> &potentialFlows) {
    constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
    constexpr auto exec_tag = wrapv<space>{};

    auto allocator = get_temporary_memory_source(pol);

    visited.reset(0);
    visited.setVal(1, source);
    frontier.reset(0);
    frontier.setVal(1, source);
    parents.setVal(-1, source);
    potentialFlows.reset(0);
    potentialFlows.setVal(std::numeric_limits<T>::max(), source);

    auto isFrontierEmpty = Vector<int>(allocator, 1);
    auto isSinkFound = Vector<int>(allocator, 1);

    while (true) {
      isFrontierEmpty.setVal(1);
      {
        auto &params = zs::make_tuple(view<space>(isFrontierEmpty));
        pol(zs::range(frontier), params, kernel_check_empty_frontier{});
      }
      if (isFrontierEmpty.getVal()) break;

      isSinkFound.setVal(0);
      {
        auto params = zs::make_tuple(view<space>(visited), view<space>(potentialFlows), sink,
                                     view<space>(parents), view<space>(frontier),
                                     view<space>(isSinkFound), proxy<space>(capacity), exec_tag);
        pol(zs::range(capacity.outerSize()), params, kernel_bfs_augmented_path{});
      }
      if (isSinkFound.getVal()) break;
    }
    return potentialFlows.getVal(sink) > 0;
  }

  struct kernel_initialize_hashtable {
    template <typename ParamT> constexpr void operator()(int vi, ParamT &params) noexcept {
      auto &[capacity, h, hashBuffer] = params;
      for (int j = capacity._ptrs[vi]; j < capacity._ptrs[vi + 1]; ++j) {
        auto nvi = capacity._inds[j];
        auto cnt = h.insert({vi, nvi});
        hashBuffer[cnt] = j;
      }
    }
  };

  template <typename Policy, typename SpMat, typename T = typename SpMat::value_type,
            typename Ti = typename SpMat::index_type,
            enable_if_all<!is_const_v<SpMat>, is_spmat_v<remove_cv_t<SpMat>>> = 0>
  inline void maximum_flow(Policy &&pol, Ti source, Ti sink, SpMat &capacity, T &res) {
    constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

    auto allocator = get_temporary_memory_source(pol);

    size_t n = capacity.outerSize();
    auto visited = Vector<int>(allocator, n);
    auto frontier = Vector<int>(allocator, n);
    auto potentialFlows = Vector<T>(allocator, n);
    auto parents = Vector<int>(allocator, n);

    res = 0;
    parents.setVal(-1, source);

    auto hash = bht<int, 2>(allocator, capacity.nnz());
    auto hashBuffer = Vector<int>(allocator, capacity.nnz());

    auto &params
        = zs::make_tuple(proxy<space>(capacity), proxy<space>(hash), proxy<space>(hashBuffer));
    pol(range(n), params, kernel_initialize_hashtable{});

    hash = hash.clone({memsrc_e::host});
    auto hashView = proxy<execspace_e::host>(hash);

    while (find_augmented_path(pol, source, sink, capacity, parents, visited, frontier,
                               potentialFlows)) {
      auto minFlow = potentialFlows.getVal(sink);
      res += minFlow;

      for (int vi = sink; parents.getVal(vi) >= 0; vi = parents.getVal(vi)) {
        int pvi = parents.getVal(vi);
        int edge_id = hashBuffer.getVal(hashView.query({pvi, vi}));
        int reverse_edge_id = hashBuffer.getVal(hashView.query({vi, pvi}));

        capacity._vals.setVal(capacity._vals.getVal(edge_id) - minFlow, edge_id);
        capacity._vals.setVal(capacity._vals.getVal(reverse_edge_id) + minFlow, reverse_edge_id);
      }
    }
  }

  /*
  inline int maximum_flow(const std::vector<std::array<int, 3>> &DAG, int nmNodes,
                          int source, int sink) {
    std::vector<std::vector<int>> topology(nmNodes);
    std::vector<std::vector<int>> capacity(nmNodes);

    for (const auto &edge_capacity : DAG) {
      topology[edge_capacity[0]].push_back(edge_capacity[1]);
      capacity[edge_capacity[0]].push_back(edge_capacity[2]);

      topology[edge_capacity[1]].push_back(edge_capacity[0]);
      capacity[edge_capacity[1]].push_back(0);
    }

    std::vector<bool> visited(nmNodes, false);
    std::vector<int> potentialFlows(nmNodes, 0);
    std::vector<int> parents(nmNodes, -1);

    int minFlow = 0;

    while (true) {
      std::fill(visited.begin(), visited.end(), false);

      std::fill(potentialFlows.begin(), potentialFlows.end(), 0);
      potentialFlows[source] = std::numeric_limits<int>::max();
      std::fill(parents.begin(), parents.end(), -1);

      std::queue<int> q;
      q.push(source);

      visited[source] = true;
      bool find_sink = false;

      while (!q.empty() && !find_sink) {
        auto node = q.front();
        q.pop();
        for (int i = 0; i != topology[node].size(); ++i) {
          int nnode = topology[node][i];
          if (visited[nnode])
            continue;
          if (capacity[node][i] == 0)
            continue;
          visited[nnode] = true;
          parents[nnode] = node;
          potentialFlows[nnode] =
              std::min(potentialFlows[node], capacity[node][i]);

          if (nnode == sink) {
            find_sink = true;
            break;
          }
          q.push(nnode);
        }
      }

      if (find_sink) {
        int sinkFlow = potentialFlows[sink];
        minFlow += sinkFlow;
        for (int node = sink; parents[node] >= 0; node = parents[node]) {
          int parentNode = parents[node];
          for (int i = 0; i != capacity[node].size(); ++i) {
            if (topology[node][i] == parentNode) {
              capacity[node][i] += sinkFlow;
              break;
            }
          }
          for (int i = 0; i != capacity[parentNode].size(); ++i) {
            if (topology[parentNode][i] == node) {
              capacity[parentNode][i] -= sinkFlow;
              break;
            }
          }

          assert(capacity[pnode][local_id] >= 0 &&
                 capacity[node][reverse_local_id] >= 0);
        }

      } else {
        break;
      }
    }

    return minFlow;
  }
  */

}  // namespace zs
