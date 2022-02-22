#include <stdexcept>

#include "LevelSetUtils.hpp"

namespace zs {

  /// max velocity
  template <typename ExecPol, int dim, grid_e category>
  void get_level_set_max_speed(ExecPol &&pol, SparseLevelSet<dim, category> &ls) {
    constexpr execspace_e space = RM_CVREF_T(pol)::exec_tag::value;

    Vector<typename SparseLevelSet<dim, category>::value_type> vel{ls.get_allocator(), 1};
    vel.setVal(0);
    if (ls.hasProperty("vel")) {
      auto nbs = ls.numBlocks();
      pol(std::initializer_list<std::size_t>{nbs, ls.block_size},
          [ls = proxy<space>(ls), vel = vel.data()] ZS_LAMBDA(
              typename RM_CVREF_T(ls)::size_type bi,
              typename RM_CVREF_T(ls)::cell_index_type ci) mutable {
            using ls_t = RM_CVREF_T(ls);
            auto coord = ls._table._activeKeys[bi] + ls_t::cellid_to_coord(ci);
            typename ls_t::TV vi{};
            if constexpr (ls_t::category == grid_e::staggered)
              vi = ls.ipack("vel", coord, 0);
            else
              vi = ls.wpack<3>("vel", ls.indexToWorld(coord), 0);
            vi = vi.abs();
            auto vm = vi[0];
            if (vi[1] > vm) vm = vi[1];
            if (vi[2] > vm) vm = vi[2];
            atomic_max(wrapv<space>{}, vel, vm);
          });
    }
    return vel.getVal();
  }

  /// mark
  template <typename ExecPol, int dim, grid_e category>
  void mark_level_set(ExecPol &&pol, SparseLevelSet<dim, category> &ls,
                      typename SparseLevelSet<dim, category>::value_type threshold
                      = limits<typename SparseLevelSet<dim, category>::value_type>::epsilon()
                        * 128) {
    auto nbs = ls.numBlocks();
    constexpr execspace_e space = RM_CVREF_T(pol)::exec_tag::value;

    ls.append_channels(pol, {{"mark", 1}});

    pol(std::initializer_list<std::size_t>{nbs, ls.block_size},
        [ls = proxy<space>(ls), threshold] ZS_LAMBDA(
            typename RM_CVREF_T(ls)::size_type bi,
            typename RM_CVREF_T(ls)::cell_index_type ci) mutable {
          using ls_t = RM_CVREF_T(ls);
          if constexpr (ls_t::category == grid_e::staggered) {
            for (typename ls_t::channel_counter_type propNo = 0; propNo != ls.numProperties();
                 ++propNo) {
              if (ls.getPropertyNames()[propNo] == "mark") continue;  // skip property ["mark"]
              auto propOffset = ls.getPropertyOffsets()[propNo];
              auto propSize = ls.getPropertySizes()[propNo];
              auto coord = ls._table._activeKeys[bi] + ls_t::cellid_to_coord(ci);
              // usually this is
              for (typename ls_t::channel_counter_type chn = 0; chn != propSize; ++chn) {
                if (zs::abs(ls.value_or(propOffset + chn, coord, chn % 3, 0)) > threshold
                    || zs::abs(ls.value_or(propOffset + chn, coord, chn % 3 + 3, 0)) > threshold) {
                  ls._grid("mark", bi, ci) = 1;
                  // atomic_add(wrapv<space>{}, cnt, (u64)1);
                  break;  // no need further checking
                }
              }
            }
          } else {
            auto block = ls._grid.block(bi);
            // const auto nchns = ls.numChannels();
            for (typename ls_t::channel_counter_type propNo = 0; propNo != ls.numProperties();
                 ++propNo) {
              if (ls.getPropertyNames()[propNo] == "mark") continue;  // skip property ["mark"]
              auto propOffset = ls.getPropertyOffsets()[propNo];
              auto propSize = ls.getPropertySizes()[propNo];
              for (typename ls_t::channel_counter_type chn = 0; chn != propSize; ++chn)
                if (zs::abs(block(propOffset + chn, ci)) > threshold) {
                  block("mark", ci) = 1;
                  // atomic_add(wrapv<space>{}, cnt, (u64)1);
                  break;  // no need further checking
                }
            }
          }
        });
  }

  /// shrink
  template <typename ExecPol, int dim, grid_e category> void refit_level_set_domain(
      ExecPol &&pol, SparseLevelSet<dim, category> &ls,
      typename SparseLevelSet<dim, category>::value_type threshold
      = zs::limits<typename SparseLevelSet<dim, category>::value_type>::epsilon() * 128) {
    using SplsT = SparseLevelSet<dim, category>;
    mark_level_set(pol, ls);

    constexpr execspace_e space = RM_CVREF_T(pol)::exec_tag::value;
    std::size_t nbs = ls.numBlocks();
    const auto &allocator = ls.get_allocator();

    // mark
    Vector<typename SplsT::size_type> marks{allocator, nbs + 1};
    pol(std::initializer_list<std::size_t>{nbs, ls.block_size},
        [ls = proxy<space>(ls), marks = proxy<space>(marks)] ZS_LAMBDA(
            typename RM_CVREF_T(ls)::size_type bi,
            typename RM_CVREF_T(ls)::cell_index_type ci) mutable {
          marks(bi) = ls._grid("mark", bi, ci) != 0 ? 1 : 0;
        });

    // establish mapping
    Vector<typename SplsT::size_type> offsets{allocator, nbs + 1};
    exclusive_scan(pol, std::begin(marks), std::end(marks), std::begin(offsets),
                   (typename SplsT::size_type)0, plus<typename SplsT::size_type>{});
    std::size_t newNbs = offsets.getVal(nbs);  // retrieve latest numblocks
    fmt::print(fg(fmt::color::blue_violet), "shrink [{}] blocks to [{}] blocks\n", nbs, newNbs);
    Vector<typename SplsT::size_type> preservedBlockNos{allocator, newNbs};
    pol(range(newNbs),
        [marks = proxy<space>(marks), offsets = proxy<space>(offsets),
         preservedBlockNos = proxy<space>(
             preservedBlockNos)] ZS_LAMBDA(typename RM_CVREF_T(ls)::size_type bi) mutable {
          if (marks[bi] != 0) preservedBlockNos[offsets[bi]] = bi;
        });

    using TableT = typename SplsT::table_t;
    using GridT = typename SplsT::grid_t;

    TableT newTable{allocator, newNbs};
    GridT newGrid{allocator, ls._grid.getProperties(), ls._grid._dx, newNbs};

    // shrink table
    pol(range(newTable._tableSize),
        zs::ResetHashTable{proxy<space>(newTable)});  // cnt not yet cleared

    pol(range(newNbs),
        [blocknos = proxy<space>(preservedBlockNos), blockids = proxy<space>(ls._table._activeKeys),
         newTable = proxy<space>(newTable),
         newNbs] ZS_LAMBDA(typename RM_CVREF_T(ls)::size_type bi) mutable {
          newTable.insert(blockids[blocknos[bi]], bi);
          if (bi == 0) *newTable._cnt = newNbs;
        });
    // shrink grid
    pol(std::initializer_list<std::size_t>{newNbs, ls.block_size},
        [blocknos = proxy<space>(preservedBlockNos), grid = proxy<space>(ls._grid),
         newGrid
         = proxy<space>(newGrid)] ZS_LAMBDA(typename RM_CVREF_T(ls)::size_type bi,
                                            typename RM_CVREF_T(ls)::cell_index_type ci) mutable {
          using grid_t = RM_CVREF_T(grid);
          const auto block = grid.block(blocknos[bi]);
          auto newBlock = newGrid.block(bi);
          for (typename grid_t::channel_counter_type chn = 0; chn != newGrid.numChannels(); ++chn)
            newBlock(chn, ci) = block(chn, ci);
        });

    ls._table = std::move(newTable);
    ls._grid = std::move(newGrid);
  }

  /// usually shrink before extend
  template <typename ExecPol, int dim, grid_e category>
  void extend_level_set_domain(ExecPol &&pol, SparseLevelSet<dim, category> &ls, int nlayers) {
    constexpr execspace_e space = RM_CVREF_T(pol)::exec_tag::value;

    while (nlayers--) {
      auto nbs = ls.numBlocks();
      if (nbs * 26 >= ls.numReservedBlocks())
        ls.resize(pol, nbs * 26);  // at most 26 neighbor blocks spawned
      pol(range(nbs), [ls = proxy<space>(ls)](typename RM_CVREF_T(ls)::size_type bi) mutable {
        using ls_t = RM_CVREF_T(ls);
        using table_t = RM_CVREF_T(ls._table);
        auto coord = ls._table._activeKeys[bi];
        for (auto loc : ndrange<3>(3)) {
          auto offset = (make_vec<int>(loc) - 1) * ls_t::side_length;
          using TV = RM_CVREF_T(offset);
          if (offset == TV::zeros()) return;
          if (auto blockno = ls._table.insert(coord + offset);
              blockno != table_t::sentinel_v) {  // initialize newly inserted block
            auto block = ls._grid.block(blockno);
            for (typename ls_t::channel_counter_type chn = 0; chn != ls.numChannels(); ++chn)
              for (typename ls_t::cell_index_type ci = 0; ci != ls.block_size; ++ci)
                block(chn, ci) = ls._backgroundValue;
          }
        }
      });
    }
  }

  template <typename ExecPol, int dim, grid_e category>
  void flood_fill_levelset(ExecPol &&policy, SparseLevelSet<dim, category> &ls) {
    constexpr execspace_e space = RM_CVREF_T(policy)::exec_tag::value;
    using SpLs = SparseLevelSet<dim, category>;

    auto &grid = ls._grid;
    auto &table = ls._table;
    auto &blocks = table._activeKeys;

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