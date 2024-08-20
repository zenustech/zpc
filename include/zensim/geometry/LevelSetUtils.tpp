#include <stdexcept>

#include "LevelSetUtils.hpp"
#include "zensim/math/MathUtils.h"

namespace zs {

  /// max velocity
  template <typename ls_t, typename T> struct max_ls_speed_op {
    constexpr void operator()(typename ls_t::size_type bi, typename ls_t::cell_index_type ci) {
      constexpr auto dim = ls_t::dim;
      constexpr auto space = ls_t::space;
      auto coord = ls._table._activeKeys[bi] + ls_t::grid_view_t::cellid_to_coord(ci);
      typename ls_t::TV vi{};
      if constexpr (ls_t::category == grid_e::staggered)
        vi = ls.ipack("v", ls.cellToIndex(coord), 0);
      else
        vi = ls.wpack<dim>("v", ls.indexToWorld(coord), 0);
      vi = vi.abs();
      auto vm = vi[0];
      for (int d = 1; d != dim; ++d)
        if (vi[d] > vm) vm = vi[d];
      atomic_max(wrapv<space>{}, vel, vm);
    }
    ls_t ls;
    T *vel;  // max velocity inf norm
  };
  template <typename LsvT, typename T> max_ls_speed_op(LsvT, T *) -> max_ls_speed_op<LsvT, T>;

  template <typename ExecPol, int dim, grid_e category>
  auto get_level_set_max_speed(ExecPol &&pol, const SparseLevelSet<dim, category> &ls) ->
      typename SparseLevelSet<dim, category>::value_type {
    constexpr execspace_e space = RM_REF_T(pol)::exec_tag::value;

    Vector<typename SparseLevelSet<dim, category>::value_type> vel{ls.get_allocator(), 1};
    vel.setVal(0);
    if (ls.hasProperty("v")) {
      auto nbs = ls.numBlocks();
      pol(Collapse{nbs, ls.block_size}, max_ls_speed_op{proxy<space>(ls), vel.data()});
#if 0
      pol(Collapse{(sint_t)nbs, (sint_t)ls.block_size},
          [ls = proxy<space>(ls), vel = vel.data()] ZS_LAMBDA(
              typename RM_REF_T(ls)::size_type bi,
              typename RM_REF_T(ls)::cell_index_type ci) mutable {
            using ls_t = RM_CVREF_T(ls);
            auto coord = ls._table._activeKeys[bi] + ls_t::grid_view_t::cellid_to_coord(ci);
            typename ls_t::TV vi{};
            if constexpr (ls_t::category == grid_e::staggered)
              vi = ls.ipack("v", ls.cellToIndex(coord), 0);
            else
              vi = ls.wpack<3>("v", ls.indexToWorld(coord), 0);
            vi = vi.abs();
            auto vm = vi[0];
            if (vi[1] > vm) vm = vi[1];
            if (vi[2] > vm) vm = vi[2];
            atomic_max(wrapv<space>{}, vel, vm);
          });
#endif
    }
    return vel.getVal();
  }

  /// mark
  template <typename ExecPol, int dim, grid_e category>
  void mark_level_set(ExecPol &&pol, SparseLevelSet<dim, category> &ls,
                      typename SparseLevelSet<dim, category>::value_type threshold
                      = detail::deduce_numeric_epsilon<typename SparseLevelSet<dim, category>::value_type>()
                        * 128) {
    auto nbs = ls.numBlocks();
    constexpr execspace_e space = RM_REF_T(pol)::exec_tag::value;

    ls.append_channels(pol, {{"mark", 1}});

    Vector<u64> numActiveVoxels{ls.get_allocator(), 1};
    numActiveVoxels.setVal(0);

    pol(Collapse{nbs, ls.block_size},
        [ls = proxy<space>(ls), threshold, cnt = numActiveVoxels.data()] ZS_LAMBDA(
            typename RM_REF_T(ls)::size_type bi,
            typename RM_REF_T(ls)::cell_index_type ci) mutable {
          using ls_t = RM_CVREF_T(ls);
          bool done = false;
          if constexpr (ls_t::category == grid_e::staggered) {
            ls._grid("mark", bi, ci) = 0;
            for (typename ls_t::channel_counter_type propNo = 0; propNo != ls.numProperties();
                 ++propNo) {
              if (ls.getPropertyNames()[propNo] == "mark") continue;  // skip property ["mark"]
              bool isSdf = ls.getPropertyNames()[propNo] == "sdf";
              auto propOffset = ls.getPropertyOffsets()[propNo];
              auto propSize = ls.getPropertySizes()[propNo];
              auto coord = ls._table._activeKeys[bi] + ls_t::grid_view_t::cellid_to_coord(ci);
              // usually this is
              for (typename ls_t::channel_counter_type chn = 0; chn != propSize; ++chn) {
                if ((!isSdf
                     && (zs::abs(ls.value_or(propOffset + chn, coord, chn % 3, 0)) > threshold
                         || zs::abs(ls.value_or(propOffset + chn, coord, chn % 3 + 3, 0))
                                > threshold))
                    || (isSdf
                        && (ls.value_or(propOffset + chn, coord, chn % 3, 0)
                                < ls._backgroundValue + threshold
                            || ls.value_or(propOffset + chn, coord, chn % 3 + 3, 0)
                                   < ls._backgroundValue + threshold))) {
                  ls._grid("mark", bi, ci) = 1;
                  atomic_add(wrapv<space>{}, cnt, (u64)1);
                  done = true;
                  break;  // no need further checking
                }
              }
              if (done) break;
            }
          } else {
            auto block = ls._grid.block(bi);
            block("mark", ci) = 0;
            // const auto nchns = ls.numChannels();
            for (typename ls_t::channel_counter_type propNo = 0; propNo != ls.numProperties();
                 ++propNo) {
              if (ls.getPropertyNames()[propNo] == "mark") continue;  // skip property ["mark"]
              bool isSdf = ls.getPropertyNames()[propNo] == "sdf";
              auto propOffset = ls.getPropertyOffsets()[propNo];
              auto propSize = ls.getPropertySizes()[propNo];
              for (typename ls_t::channel_counter_type chn = 0; chn != propSize; ++chn)
                if ((zs::abs(block(propOffset + chn, ci)) > threshold && !isSdf)
                    || (block(propOffset + chn, ci) < ls._backgroundValue + threshold && isSdf)) {
                  block("mark", ci) = 1;
                  atomic_add(wrapv<space>{}, cnt, (u64)1);
                  done = true;
                  break;  // no need further checking
                }
              if (done) break;
            }
          }
        });

    fmt::print("{} voxels ot of {} in total are active. (threshold: {})\n",
               numActiveVoxels.getVal(), ls.numBlocks() * (size_t)ls.block_size, threshold);
  }

  /// shrink
  template <typename ExecPol, int dim, grid_e category> void refit_level_set_domain(
      ExecPol &&pol, SparseLevelSet<dim, category> &ls,
      typename SparseLevelSet<dim, category>::value_type threshold
      = zs::detail::deduce_numeric_epsilon<typename SparseLevelSet<dim, category>::value_type>() * 128) {
    using SplsT = SparseLevelSet<dim, category>;

    constexpr execspace_e space = RM_REF_T(pol)::exec_tag::value;
    size_t nbs = ls.numBlocks();
    const auto &allocator = ls.get_allocator();

#if 0
    {
      Vector<float> test{ls.get_allocator(), 1};
      test.setVal(0);
      pol(range(nbs), [grid = proxy<space>(ls._grid), test = test.data()] ZS_LAMBDA(
                          typename RM_REF_T(ls)::size_type bi) mutable {
        using grid_t = RM_CVREF_T(grid);
        const auto block = grid.block(bi);
        if (!block.hasProperty("sdf")) return;
        for (int ci = 0; ci != grid.block_size; ++ci)
          if (block("sdf", ci) < 0) atomic_add(wrapv<space>{}, test, block("sdf", ci));
      });
      fmt::print("before refit, {} blocks sdf sum: {}\n", nbs, test.getVal());
    }
#endif

    mark_level_set(pol, ls, threshold);

    // mark
    Vector<typename SplsT::size_type> marks{allocator, nbs + 1};
    pol(range(nbs), [marks = proxy<space>(marks)] ZS_LAMBDA(typename RM_CVREF_T(ls)::size_type bi) {
      marks[bi] = 0;
    });
    pol(Collapse{nbs, ls.block_size},
        [ls = proxy<space>(ls), marks = proxy<space>(marks)] ZS_LAMBDA(
            typename RM_REF_T(ls)::size_type bi,
            typename RM_REF_T(ls)::cell_index_type ci) mutable {
          if ((int)ls._grid("mark", bi, ci) != 0) marks[bi] = 1;
        });

    // establish mapping
    Vector<typename SplsT::size_type> offsets{allocator, nbs + 1};
    exclusive_scan(pol, std::begin(marks), std::end(marks), std::begin(offsets));
    size_t newNbs = offsets.getVal(nbs);  // retrieve latest numblocks
    fmt::print(fg(fmt::color::blue_violet), "shrink [{}] blocks to [{}] blocks\n", nbs, newNbs);
    Vector<typename SplsT::size_type> preservedBlockNos{allocator, newNbs};
    pol(range(nbs),
        [marks = proxy<space>(marks), offsets = proxy<space>(offsets),
         blocknos
         = proxy<space>(preservedBlockNos)] ZS_LAMBDA(typename RM_REF_T(ls)::size_type bi) mutable {
          if (marks[bi] != 0) blocknos[offsets[bi]] = bi;
        });

    auto prevKeys = ls._table._activeKeys.clone(allocator);
    auto prevGrid = ls._grid.clone(allocator);
    // ls.resize(pol, newNbs);
    // ls._table.preserve(pol, newNbs);

#if 1
    // shrink table
    ls._table.reset(false);  // do not clear cnt
    pol(range(newNbs), [blocknos = proxy<space>(preservedBlockNos),
                        blockids = proxy<space>(prevKeys), newTable = proxy<space>(ls._table),
                        newNbs] ZS_LAMBDA(typename RM_REF_T(ls)::size_type bi) mutable {
      auto blockid = blockids[blocknos[bi]];
      newTable.insert(blockid, bi);
      newTable._activeKeys[bi] = blockid;
      if (bi == 0) *newTable._cnt = newNbs;
    });
// shrink grid
#  if 0
    {
      Vector<float> test{ls.get_allocator(), 1};
      test.setVal(0);
      pol(range(nbs), [grid = proxy<space>(ls), test = test.data()] ZS_LAMBDA(
                          typename RM_REF_T(ls)::size_type bi) mutable {
        using grid_t = RM_CVREF_T(grid);
        const auto block = grid.block(bi);
        if (!block.hasProperty("sdf")) return;
        for (int ci = 0; ci != grid.block_size; ++ci)
          if (block("sdf", ci) < 0) atomic_add(wrapv<space>{}, test, block("sdf", ci));
      });
      fmt::print("before grid built, {} blocks sdf sum: {}\n", nbs, test.getVal());
    }
#  endif
    // Vector<float> test{ls.get_allocator(), 1};
    // test.setVal(0);
    pol(Collapse{newNbs, ls.block_size},
        [blocknos = proxy<space>(preservedBlockNos), grid = proxy<space>(prevGrid),
         marks = proxy<space>(marks), offsets = proxy<space>(offsets),
         ls = proxy<space>(ls)] ZS_LAMBDA(typename RM_REF_T(ls)::size_type bi,
                                          typename RM_REF_T(ls)::cell_index_type ci) mutable {
          using grid_t = RM_CVREF_T(grid);
          const auto block = grid.block(blocknos[bi]);
          auto newBlock = ls._grid.block(bi);
          for (typename grid_t::channel_counter_type chn = 0; chn != ls.numChannels(); ++chn)
            newBlock(chn, ci) = block(chn, ci);
          // if (ls.hasProperty("sdf"))
          //  if (newBlock("sdf", ci) < 0) atomic_add(wrapv<space>{}, test, newBlock("sdf", ci));
        });
#  if 0
    auto newSum = test.getVal();
    test.setVal(0);
    pol(range(nbs), [grid = proxy<space>(prevGrid), marks = proxy<space>(marks),
                     offsets = proxy<space>(offsets),
                     test = test.data()] ZS_LAMBDA(typename RM_CVREF_T(ls)::size_type bi) mutable {
      // if (marks[bi] == 1)
      //  printf("@@ [%d] mark{%d}, offset{%d}\n", (int)bi, (int)marks[bi], (int)offsets[bi]);
      using grid_t = RM_CVREF_T(grid);
      const auto block = grid.block(bi);
      if (!block.hasProperty("sdf")) return;
      for (int ci = 0; ci != grid.block_size; ++ci)
        if (block("sdf", ci) < -1e-6) atomic_add(wrapv<space>{}, test, block("sdf", ci));
    });
    auto totalSum = test.getVal();
    test.setVal(0);
    pol(range(nbs), [grid = proxy<space>(prevGrid), marks = proxy<space>(marks),
                     offsets = proxy<space>(offsets),
                     test = test.data()] ZS_LAMBDA(typename RM_REF_T(ls)::size_type bi) mutable {
      if (marks[bi] == 0) return;
      using grid_t = RM_CVREF_T(grid);
      const auto block = grid.block(bi);
      if (!block.hasProperty("sdf")) return;
      for (int ci = 0; ci != grid.block_size; ++ci)
        if (block("sdf", ci) < -1e-6) atomic_add(wrapv<space>{}, test, block("sdf", ci));
    });
    fmt::print("\tsummed sdf [old] {} (marked {}); [new] {}\n", totalSum, test.getVal(), newSum);
    getchar();
#  endif
#endif
  }

  /// usually shrink before extend
  template <typename ExecPol, int dim, grid_e category>
  void extend_level_set_domain(ExecPol &&pol, SparseLevelSet<dim, category> &ls, int nlayers) {
    constexpr execspace_e space = RM_REF_T(pol)::exec_tag::value;

    constexpr auto coeff = math::pow_integral(dim, dim) - 1;
    auto nbs = ls.numBlocks();
    typename RM_REF_T(ls)::size_type base = 0;
    // [base, nbs): candidate blocks to expand from
    // [nbs, newNbs): newly spawned
    while (nlayers--) {
      if (auto expectedNum = (nbs - base) * coeff + nbs; expectedNum >= ls.numReservedBlocks()) {
        fmt::print("reserving levelset from {} to {} blocks\n", ls.numReservedBlocks(),
                   expectedNum);
        ls.resize(pol, expectedNum);
      }
      pol(range(nbs - base),
          [ls = proxy<space>(ls), base] ZS_LAMBDA(typename RM_REF_T(ls)::size_type bi) mutable {
            bi += base;
            using ls_t = RM_CVREF_T(ls);
            using table_t = RM_CVREF_T(ls._table);
            auto coord = ls._table._activeKeys[bi];
#if 0
            bool active = false;
            for (typename ls_t::cell_index_type ci = 0; ci != ls.block_size; ++ci)
              if ((int)ls._grid("mark", bi, ci) != 0) {
                active = true;
                break;
              }
            if (!active) return;
#endif
            for (auto loc : ndrange<3>(3)) {
              auto offset = (make_vec<int>(loc) - 1) * (typename ls_t::size_type)ls_t::side_length;
              ls._table.insert((coord + offset).template cast<int>());
            }
          });
      auto newNbs = ls.numBlocks();
      ls._grid.resize(newNbs);
      pol(Collapse{newNbs - nbs, ls.block_size},
          [ls = proxy<space>(ls), nbs] ZS_LAMBDA(
              typename RM_REF_T(ls)::size_type bi,
              typename RM_REF_T(ls)::cell_index_type ci) mutable {
            using ls_t = RM_CVREF_T(ls);
            using table_t = RM_CVREF_T(ls._table);
            auto block = ls._grid.block(bi + nbs);
            for (typename ls_t::channel_counter_type chn = 0; chn != ls.numChannels(); ++chn)
              block(chn, ci) = 0;
            if (ls.hasProperty("sdf"))
              block("sdf", ci) = ls._backgroundValue + ls._grid.dx;  // trick 'mark_activation'
          });
      fmt::print("{} blocks being spawned to {} blocks.\n", nbs, newNbs);
      base = nbs;
      nbs = newNbs;
    }
  }

  template <typename ExecPol, int dim, grid_e category, typename SdfLsvT>
  void extend_level_set_domain(ExecPol &&pol, SparseLevelSet<dim, category> &ls,
                               const SdfLsvT sdfLsv) {
    static_assert(dim == 3, "currently only supports 3d");
    using ls_t = RM_CVREF_T(ls);
    constexpr execspace_e space = RM_REF_T(pol)::exec_tag::value;

    ls.append_channels(pol, {{"mark", 1}});
    auto markOffset = ls._grid.getPropertyOffset("mark");
    auto nbs = ls.numBlocks();
    /// mark all existing cells
    pol(Collapse{nbs, ls.block_size},
        [ls = proxy<space>(ls), markOffset] ZS_LAMBDA(typename ls_t::size_type bi,
                                                      typename ls_t::cell_index_type ci) mutable {
          ls._grid(markOffset, bi, ci) = 1;
        });

    constexpr auto coeff = math::pow_integral(dim, dim) - 1;
    typename ls_t::size_type base = 0;
    // [base, nbs): candidate blocks to expand from
    // [nbs, newNbs): newly spawned
    while (true) {
      if (auto expectedNum = (nbs - base) * coeff + nbs; expectedNum >= ls.numReservedBlocks()) {
        fmt::print("reserving levelset from {} to {} blocks\n", ls.numReservedBlocks(),
                   expectedNum);
        ls.resize(pol, expectedNum);
      }
      pol(Collapse{nbs - base, ls.block_size},
          [ls = proxy<space>(ls), sdfLsv, base] ZS_LAMBDA(
              typename ls_t::size_type bi, typename ls_t::cell_index_type ci) mutable {
            using lsv_t = RM_CVREF_T(ls);
            bi += base;
            using table_t = RM_CVREF_T(ls._table);
            auto coord = ls._table._activeKeys[bi] + lsv_t::grid_view_t::cellid_to_coord(ci);
            for (auto loc : ndrange<dim>(3)) {
              auto c = coord + (make_vec<int>(loc) - 1);
              auto cb = c - (c & (lsv_t::side_length - 1));
              if (ls._table.query(cb) == table_t::sentinel_v
                  && sdfLsv.getSignedDistance(ls.indexToWorld(c)) < 0)
                ls._table.insert(cb);
            }
          });
      auto newNbs = ls.numBlocks();
      fmt::print("[extend_level_set_domain]: looking at expanding blocks from {} -> {}\n", nbs,
                 newNbs);
      if (newNbs == nbs) break;

      ls._grid.resize(newNbs);
      pol(Collapse{newNbs - nbs, ls.block_size},
          [ls = proxy<space>(ls), nbs] ZS_LAMBDA(typename ls_t::size_type bi,
                                                 typename ls_t::cell_index_type ci) mutable {
            ls._grid("mark", bi + nbs, ci) = 0;
          });
      pol(Collapse{newNbs - nbs, ls.block_size},
          [ls = proxy<space>(ls), nbs, markOffset] ZS_LAMBDA(
              typename ls_t::size_type bi, typename ls_t::cell_index_type ci) mutable {
            using lsv_t = RM_CVREF_T(ls);
            bi += nbs;
            using table_t = RM_CVREF_T(ls._table);
            auto coord = ls._table._activeKeys[bi] + lsv_t::grid_view_t::cellid_to_coord(ci);

            auto block = ls._grid.block(bi);
            int iter = 0;
            bool done = false;
            for (; iter != ls.side_length; ++iter) {
              if (!done) {
                /// first accumulate num of active cells
                int numActiveNei = 0;
                i32 flag = 0;
                for (auto loc : ndrange<dim>(3)) {
                  auto c = coord + (make_vec<int>(loc) - 1);
                  // auto cb = c - (c & (ls_t::side_length - 1));
                  if ((int)ls.value_or(markOffset, c, 0) != 0) {
                    numActiveNei += 1;
                    flag |= 1 << (get<0>(loc) * 9 + get<1>(loc) * 3 + get<2>(loc));
                  }
                }
                if (numActiveNei != 0) {
                  done = true;
                  ls._grid("mark", bi, ci) = 1;
                  //
                  for (typename ls_t::channel_counter_type propNo = 0; propNo != ls.numProperties();
                       ++propNo) {
                    // skip property ["mark", "sdf"]
                    if (ls.getPropertyNames()[propNo] == "mark"
                        || ls.getPropertyNames()[propNo] == "sdf")
                      continue;
                    auto propOffset = ls.getPropertyOffsets()[propNo];
                    auto propSize = ls.getPropertySizes()[propNo];
                    for (typename ls_t::channel_counter_type chn = 0; chn != propSize; ++chn) {
                      typename ls_t::value_type sum = 0;
                      for (auto loc : ndrange<3>(3))
                        if (flag & (1 << (get<0>(loc) * 9 + get<1>(loc) * 3 + get<2>(loc)))) {
                          auto c = coord + (make_vec<int>(loc) - 1);
                          sum += ls.value_or(propOffset + chn, c, 0);
                        }
                      ls._grid(propOffset + chn, bi, ci) = sum / numActiveNei;
                    }
                  }
                }
              }
              sync_threads(wrapv<space>{});
            }
          });
      fmt::print("{} blocks being spawned to {} blocks.\n", nbs, newNbs);
      base = nbs;
      nbs = newNbs;
    }
  }

  template <typename ExecPol, int dim, grid_e category>
  void flood_fill_levelset(ExecPol &&policy, SparseLevelSet<dim, category> &ls) {
    constexpr execspace_e space = RM_REF_T(policy)::exec_tag::value;
    using SpLs = SparseLevelSet<dim, category>;

    auto &grid = ls._grid;
    auto &table = ls._table;
    auto &blocks = table._activeKeys;

    if (!ls.hasProperty("mask")) throw std::runtime_error("missing mask info in the levelset!");
    std::vector<PropertyTag> tags{};
    if (!ls.hasProperty("tag")) tags.push_back({"tag", 1});
    if (!ls.hasProperty("tagmask")) tags.push_back({"tagmask", 1});
    if (tags.size()) {
      ls.append_channels(policy, tags);
      policy(range(grid.size()),
             InitFloodFillGridChannels<decltype(proxy<space>(grid))>{proxy<space>(grid)});
      fmt::print("tagmask at chn {}, tag at chn {}\n", grid.getPropertyOffset("tagmask"),
                 grid.getPropertyOffset("tag"));
    }
    fmt::print("sdf at chn {}, mask at chn {}\n", grid.getPropertyOffset("sdf"),
               grid.getPropertyOffset("mask"));

    fmt::print(
        "block capacity: {}, table block count: {}, cell count: {} ({}), tag chn offset: {}\n",
        blocks.size(), table.size(), blocks.size() * grid.block_space(), grid.size(),
        grid.getPropertyOffset("tag"));

    size_t tableSize = table.size();
    int iter = 0;
    grid.resize(tableSize * 2);

    Vector<typename SpLs::size_type> ks{grid.get_allocator(), grid.size()};
    do {
      Vector<int> tmp{1, memsrc_e::host, -1};
      tmp[0] = 0;
      auto seedcnt = tmp.clone(grid.get_allocator());

      policy(range(tableSize * (size_t)grid.block_space()),
             ReserveForNeighbor<space, RM_CVREF_T(ls)>{ls});
#if 0
      {
        auto lsv = proxy<space>(ls);
        lsv.print();
      }
      puts("done expansion");
      getchar();
#endif

      ks.resize(tableSize * (size_t)grid.block_space());
      policy(range(tableSize * (size_t)grid.block_space()),
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