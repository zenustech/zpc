#include <stdexcept>

#include "LevelSetUtils.hpp"

namespace zs {

  template <typename ExecPol, int dim, grid_e category>
  void flood_fill_levelset(ExecPol&& policy, SparseLevelSet<dim, category>& ls) {
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    using SpLs = SparseLevelSet<dim, category>;

#if 0
     if (!ls._grid.hasProperty("tag")) {
	    ls._grid.blocks.append_channels(policy, {{"tag", 1}});
    policy(range(grid.size()), FillGridChannels<RM_CVREF_T(proxy<space>(grid))>{
                                   proxy<space>(grid), grid.getChannelOffset("tag"), 0, 1});
  }
#endif
    if (!ls._grid.hasProperty("tag")) {
      throw std::runtime_error("missing initial tag info in the levelset!");
    }
    auto& grid = ls._grid;
    auto& table = ls._table;
    auto& blocks = table._activeKeys;

    fmt::print(
        "block capacity: {}, table block count: {}, cell count: {} ({}), tag chn offset: {}\n",
        blocks.size(), table.size(), blocks.size() * grid.block_space(), grid.size(),
        grid.getChannelOffset("tag"));

    std::size_t tableSize = table.size();
    int iter = 0;
    grid.resize(tableSize * 2);

    do {
      Vector<int> tmp{1, memsrc_e::host, -1};
      tmp[0] = 0;
      auto changed = tmp.clone(grid.allocator());

      policy(range(tableSize * (std::size_t)grid.block_space()),
             MarkInteriorTag<space, RM_CVREF_T(ls)>{ls, changed.data()});

      // fmt::print("floodfill iter [{}]: {} -> {}\n", iter++, tableSize, table.size());

      tableSize = table.size();
      grid.resize(tableSize * 2);

      policy(range(tableSize * (std::size_t)grid.block_space()),
             ComputeTaggedSDF<space, RM_CVREF_T(ls)>{ls, changed.data()});

      if (changed.clone(MemoryLocation{memsrc_e::host, -1})[0] == 0) break;
      // getchar();
      iter++;
    } while (true);

    fmt::print("floodfill finished at iter [{}] with {} blocks\n", iter, table.size());
    return;
  }

}  // namespace zs