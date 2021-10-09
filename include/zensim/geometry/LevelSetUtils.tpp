#include <stdexcept>

#include "LevelSetUtils.hpp"

namespace zs {

  template <typename ExecPol, int dim, grid_e category>
  void flood_fill_levelset(ExecPol&& policy, SparseLevelSet<dim, category>& ls) {
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    using SpLs = SparseLevelSet<dim, category>;

    auto& grid = ls._grid;
    auto& table = ls._table;
    auto& blocks = table._activeKeys;

    if (!ls._grid.hasProperty("mask"))
      throw std::runtime_error("missing mask info in the levelset!");
    std::vector<PropertyTag> tags{};
    if (!ls._grid.hasProperty("tag")) tags.push_back({"tag", 1});
    if (!ls._grid.hasProperty("tagmask")) tags.push_back({"tagmask", 1});
    if (tags.size()) {
      ls._grid.blocks.append_channels(policy, tags);
      policy(range(grid.size()),
             InitFloodFillGridChannels<RM_CVREF_T(proxy<space>(grid))>{proxy<space>(grid)});
      fmt::print("tagmask at chn {}, tag at chn {}\n", grid.getChannelOffset("tagmask"),
                 grid.getChannelOffset("tag"));
    }
    fmt::print("sdf at chn {}, mask at chn {}\n", grid.getChannelOffset("sdf"),
               grid.getChannelOffset("mask"));

    fmt::print(
        "block capacity: {}, table block count: {}, cell count: {} ({}), tag chn offset: {}\n",
        blocks.size(), table.size(), blocks.size() * grid.block_space(), grid.size(),
        grid.getChannelOffset("tag"));

    std::size_t tableSize = table.size();
    int iter = 0;
    grid.resize(tableSize * 2);

    Vector<typename SpLs::size_type> ks{grid.get_allocator(), grid.size()};
    do {
      Vector<int> tmp{1, memsrc_e::host, -1};
      tmp[0] = 0;
      auto seedcnt = tmp.clone(grid.get_allocator());

      policy(range(tableSize * (std::size_t)grid.block_space()),
             ReserveForNeighbor<space, RM_CVREF_T(ls)>{ls});
#if 0
      {
        auto lsv = proxy<space>(ls);
        lsv.print();
      }
      puts("done expansion");
      getchar();
#endif

      ks.resize(tableSize * (std::size_t)grid.block_space());
      policy(range(tableSize * (std::size_t)grid.block_space()),
             MarkInteriorTag<space, RM_CVREF_T(ls), RM_CVREF_T(ks)>{ls, ks, seedcnt.data()});
      tmp = seedcnt.clone({memsrc_e::host, -1});
      fmt::print("floodfill iter [{}]: {} -> {}, {} seeds\n", iter, tableSize, table.size(),
                 tmp[0]);
      if (tmp[0] == 0) break;
#if 0
      {
        auto lsv = proxy<space>(ls);
        lsv.print();
      }
      puts("done tagging");
      getchar();
#endif

      tableSize = table.size();
      grid.resize(tableSize * 2);

      policy(range(tmp[0]), ComputeTaggedSDF<space, RM_CVREF_T(ls), RM_CVREF_T(ks)>{ls, ks});

#if 0
      {
        auto lsv = proxy<space>(ls);
        lsv.print();
      }
      puts("done sdf compute");
      getchar();
#endif
      iter++;
    } while (true);

    fmt::print("floodfill finished at iter [{}] with {} blocks\n", iter, table.size());
    return;
  }

}  // namespace zs