#include "LevelSetUtils.hpp"

namespace zs {

  template <typename ExecPol, int dim, grid_e category>
  void flood_fill_levelset(ExecPol&& policy, SparseLevelSet<dim, category>& ls) {
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    using SpLs = SparseLevelSet<dim, category>;

    if (!ls._grid.hasProperty("tag")) ls._grid.blocks.append_channels(policy, {{"tag", 1}});
    auto& grid = ls._grid;
    auto& table = ls._table;
    auto& blocks = table._activeKeys;

    fmt::print("block count: {}, cell count: {}, tag chn offset: {}\n", blocks.size(),
               blocks.size() * grid.block_space(), grid.getChannelOffset("tag"));

    policy(range(blocks.size() * grid.block_space()),
           FillGridChannels<RM_CVREF_T(proxy<space>(grid))>{proxy<space>(grid),
                                                            grid.getChannelOffset("tag"), 0, 1});

    return;
  }

}  // namespace zs