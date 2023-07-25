#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/VolumeToMesh.h>

#include "VdbLevelSet.h"
#include "zensim/Logger.hpp"
#include "zensim/execution/Concurrency.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

namespace zs {

  template <typename TreeT> struct VdbConverter {
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename LeafT::ValueType;
    static_assert(RootT::LEVEL == 3, "expects a tree of 3 levels (excluding root level)");
    using ZSGridT = zs::AdaptiveGrid<3, ValueT, RootT::NodeChainType::template Get<0>::LOG2DIM,
                                     RootT::NodeChainType::template Get<1>::LOG2DIM,
                                     RootT::NodeChainType::template Get<2>::LOG2DIM>;
    using ZSCoordT = zs::vec<int, 3>;

    VdbConverter(const std::vector<unsigned int> &nodeCnts)
        : cnts(nodeCnts), ag(std::make_shared<ZSGridT>()), success{std::make_shared<bool>()} {
      // for each level, initialize allocations
      using namespace zs;
      // fmt::print("nodes per level: {}, {}, {}\n", cnts[0], cnts[1], cnts[2]);
      ag->level(dim_c<0>) = RM_CVREF_T(ag->level(dim_c<0>))({{"sdf", 1}}, nodeCnts[0]);
      ag->level(dim_c<1>) = RM_CVREF_T(ag->level(dim_c<1>))({{"sdf", 1}}, nodeCnts[1]);
      ag->level(dim_c<2>) = RM_CVREF_T(ag->level(dim_c<2>))({{"sdf", 1}}, nodeCnts[2]);
      *success = true;
    }
    ~VdbConverter() = default;

    void operator()(RootT &rt) const { ag->_background = rt.background(); }

    template <typename NodeT> void operator()(NodeT &node) const {
      using namespace zs;
      auto &level = ag->level(dim_c<NodeT::LEVEL>);
      using LevelT = RM_CVREF_T(level);
      if (auto tmp = node.getValueMask().countOn(); tmp != 0) {
        fmt::print("node [{}, {}, {}] has {} active values.\n", node.origin()[0], node.origin()[1],
                   node.origin()[2], tmp);
        *success = false;
        return;
      }
      auto &table = level.table;
      auto &grid = level.grid;
      auto &valueMask = level.valueMask;
      auto &childMask = level.childMask;
      auto tabv = proxy<execspace_e::openmp>(table);
      auto gridv = proxy<execspace_e::openmp>(grid);

      auto coord_ = node.origin();
      ZSCoordT coord{coord_[0], coord_[1], coord_[2]};
      auto bno = tabv.insert(coord);
      if (bno < 0 || bno >= cnts[NodeT::LEVEL]) {
        fmt::print("there are redundant threads inserting the same block ({}, {}, {}).\n", coord[0],
                   coord[1], coord[2]);
        *success = false;
        return;
      }

      auto block = gridv.tile(bno);
      for (auto it = node.cbeginValueOn(); it; ++it) {
        block(0, it.pos()) = it.getValue();
      }

      static_assert(
          sizeof(typename NodeT::NodeMaskType) == sizeof(typename LevelT::mask_type::value_type),
          "???");

      std::memcpy(&childMask[bno], &node.getChildMask(), sizeof(childMask[bno]));
      std::memcpy(&valueMask[bno], &node.getValueMask(), sizeof(valueMask[bno]));
    }

    void operator()(LeafT &lf) const {
#if ZS_ENABLE_OPENMP
      constexpr auto space = execspace_e::openmp;
#else
      constexpr auto space = execspace_e::host;
#endif
      using namespace zs;
      auto &level = ag->level(dim_c<0>);
      using LevelT = RM_CVREF_T(level);
      static_assert(LeafT::NUM_VALUES == ZSGridT::block_sizes[0], "????");

      auto &table = level.table;
      auto &grid = level.grid;
      auto &valueMask = level.valueMask;
      auto tabv = proxy<space>(table);
      auto gridv = proxy<space>(grid);

      auto coord_ = lf.origin();
      ZSCoordT coord{coord_[0], coord_[1], coord_[2]};
      auto bno = tabv.insert(coord);
      if (bno < 0 || bno >= cnts[0]) {
        fmt::print("there are redundant threads inserting the same leaf block ({}, {}, {}).\n",
                   coord[0], coord[1], coord[2]);
        *success = false;
        return;
      }

      auto block = gridv.tile(bno);
      for (auto it = lf.cbeginValueOn(); it; ++it) {
        block(0, it.pos()) = it.getValue();
      }

      static_assert(
          sizeof(typename LeafT::NodeMaskType) == sizeof(typename LevelT::mask_type::value_type),
          "???");
      std::memcpy(&valueMask[bno], &lf.getValueMask(), sizeof(valueMask[bno]));
    }

    ZSGridT &&get() { return std::move(*ag); }

    std::vector<unsigned int> cnts;
    std::shared_ptr<ZSGridT> ag;
    std::shared_ptr<bool> success;
  };

  /// floatgrid -> sparse grid
  AdaptiveGrid<3, f32, 3, 4, 5> convert_floatgrid_to_adaptive_grid(const OpenVDBStruct &grid,
                                                                   SmallString propTag) {
    using GridType = openvdb::FloatGrid;
    using TreeType = GridType::TreeType;
    using RootType = TreeType::RootNodeType;  // level 3 RootNode
    assert(RootType::LEVEL == 3);
    using Int1Type = RootType::ChildNodeType;  // level 2 InternalNode
    using Int2Type = Int1Type::ChildNodeType;  // level 1 InternalNode
    using LeafType = TreeType::LeafNodeType;   // level 0 LeafNode
    using LeafCIterRange = openvdb::tree::IteratorRange<typename TreeType::LeafCIter>;
    using SDFPtr = typename GridType::Ptr;
    const SDFPtr &sdf = grid.as<SDFPtr>();
    using AgT = AdaptiveGrid<3, f32, 3, 4, 5>;
    using IV = typename AgT::integer_coord_type;
    using TV = typename AgT::packed_value_type;

    using Adapter = openvdb::TreeAdapter<GridType>;
    using TreeT = typename Adapter::TreeType;
    auto &tree = Adapter::tree(*sdf);

    std::vector<unsigned int> nodeCnts(4);
    tree.root().nodeCount(nodeCnts);
    VdbConverter<TreeT> agBuilder(nodeCnts);

    // build tree
    openvdb::tree::NodeManager<TreeT> nm(sdf->tree());
    nm.foreachBottomUp(agBuilder);
    AgT ag = agBuilder.get();

    // build grid
    openvdb::Mat4R v2w = sdf->transform().baseMap()->getAffineMap()->getMat4();
    // sdf->transform().print();
    vec<f32, 4, 4> lsv2w;
    for (auto &&[r, c] : ndrange<2>(4)) lsv2w(r, c) = v2w[r][c];
    ag.resetTransformation(lsv2w);

    {
      fmt::print("nodes per level: {}, {}, {}. background: {}. dx: {}\n", nodeCnts[0], nodeCnts[1],
                 nodeCnts[2], ag._background, ag.voxelSize()[0]);
    }

    return ag;
  }
  AdaptiveGrid<3, f32, 3, 4, 5> convert_floatgrid_to_adaptive_grid(const OpenVDBStruct &grid,
                                                                   const MemoryHandle mh,
                                                                   SmallString propTag) {
    return convert_floatgrid_to_adaptive_grid(grid, propTag).clone(mh);
  }

  /// adaptive grid -> floatgrid
  OpenVDBStruct convert_adaptive_grid_to_floatgrid(const AdaptiveGrid<3, f32, 3, 4, 5> &agIn,
                                                   SmallString propTag, u32 gridClass,
                                                   SmallString gridName) {
#if ZS_ENABLE_OPENMP
    constexpr auto space = execspace_e::openmp;
    auto pol = omp_exec();
#else
    constexpr auto space = execspace_e::host;
    auto pol = seq_exec();
#endif
    using ZSCoordT = zs::vec<int, 3>;
    using GridType = openvdb::FloatGrid;
    using TreeType = GridType::TreeType;
    using RootType = TreeType::RootNodeType;   // level 3 RootNode
    using Int2Type = RootType::ChildNodeType;  // level 2 InternalNode
    using Int1Type = Int2Type::ChildNodeType;  // level 1 InternalNode
    using LeafType = TreeType::LeafNodeType;   // level 0 LeafNode
    using ValueType = LeafType::ValueType;

    const auto &ag = agIn.memspace() != memsrc_e::host ? agIn.clone({memsrc_e::host, -1}) : agIn;
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(/*background value=*/ag._background);
    // meta
    grid->insertMeta("zpc_version", openvdb::FloatMetadata(0.f));
    grid->setName(std::string(gridName));
    grid->setGridClass(static_cast<openvdb::GridClass>(gridClass));
    // transform
    openvdb::Mat4R v2w{};
    auto lsv2w = ag.getIndexToWorldTransformation();
    for (auto &&[r, c] : ndrange<2>(4)) v2w[r][c] = lsv2w[r][c];
    grid->setTransform(openvdb::math::Transform::createLinearTransform(v2w));

    // build leaf
    auto &l0 = ag.level(dim_c<0>);
    auto nlvs = l0.numBlocks();
    zs::tuple<std::vector<LeafType *>, std::vector<Int1Type *>, std::vector<Int2Type *>> nodes;
    zs::get<0>(nodes).resize(nlvs);
    pol(enumerate(l0.originRange(), zs::get<0>(nodes)),
        [grid = proxy<space>(l0.grid), vms = proxy<space>(l0.valueMask)] ZS_LAMBDA(
            size_t i, const auto &origin, LeafType *&pleaf) {
          pleaf = new LeafType();
          LeafType &leaf = const_cast<LeafType &>(*pleaf);
          leaf.setOrigin(openvdb::Coord{origin[0], origin[1], origin[2]});
          typename LeafType::NodeMaskType vm;
          std::memcpy(&vm, &vms[i], sizeof(vm));
          static_assert(sizeof(vm) == sizeof(vms[i]), "???");
          leaf.setValueMask(vm);

          auto block = grid.tile(i);
          static_assert(LeafType::SIZE == RM_CVREF_T(grid)::lane_width, "???");
          int src = 0;
          for (ValueType *dst = leaf.buffer().data(), *end = dst + LeafType::SIZE; dst != end;
               dst += 4, src += 4) {
            dst[0] = block(0, src);
            dst[1] = block(0, src + 1);
            dst[2] = block(0, src + 2);
            dst[3] = block(0, src + 3);
          }
        });

    auto build_internal = [&](auto lno) {
      constexpr int levelno = RM_CVREF_T(lno)::value;
      static_assert(levelno == 1 || levelno == 2, "???");
      using CurrentNodeType = conditional_t<levelno == 1, Int1Type, Int2Type>;
      using ChildNodeType = typename CurrentNodeType::ChildNodeType;

      auto &li = ag.level(dim_c<levelno>);
      auto &lc = ag.level(dim_c<levelno - 1>);
      auto nInts = li.numBlocks();
      zs::get<levelno>(nodes).resize(nInts);

      // @note use the child table, not from this level
      pol(enumerate(li.originRange(), zs::get<levelno>(nodes)),
          [grid = proxy<space>(li.grid), cms = proxy<space>(li.childMask),
           vms = proxy<space>(li.valueMask), tb = proxy<space>(lc.table),
           &childNodes = zs::get<levelno - 1>(nodes)](size_t i, const auto &origin,
                                                      CurrentNodeType *&pnode) mutable {
            pnode = new CurrentNodeType();
            CurrentNodeType &node = const_cast<CurrentNodeType &>(*pnode);
            auto bcoord = openvdb::Coord{origin[0], origin[1], origin[2]};
            node.setOrigin(bcoord);

            typename CurrentNodeType::NodeMaskType m;
            static_assert(sizeof(m) == sizeof(vms[i]) && sizeof(m) == sizeof(cms[i]), "???");
            std::memcpy(&m, &vms[i], sizeof(m));
            const_cast<typename CurrentNodeType::NodeMaskType &>(node.getValueMask()) = m;
            // node.setValueMask(m);
            std::memcpy(&m, &cms[i], sizeof(m));
            // node.setChildMask(m);
            const_cast<typename CurrentNodeType::NodeMaskType &>(node.getChildMask()) = m;

            auto block = grid.tile(i);
            auto *dstTable = const_cast<typename CurrentNodeType::UnionType *>(node.getTable());

            for (u32 n = 0; n < CurrentNodeType::NUM_VALUES; ++n) {
              if (m.isOn(n)) {
                auto childCoord = node.offsetToGlobalCoord(n);
                auto chNo = tb.query(ZSCoordT{childCoord[0], childCoord[1], childCoord[2]});

                ChildNodeType *chPtr = const_cast<ChildNodeType *>(childNodes[chNo]);
                dstTable[n].setChild(chPtr);
              } else {
                dstTable[n].setValue(block(0, n));
              }
            }
          });
    };

    build_internal(wrapv<1>{});
    build_internal(wrapv<2>{});

    auto &l2 = ag.level(dim_c<2>);
    auto nInt2s = l2.numBlocks();
    const auto &int2s = zs::get<2>(nodes);
    auto &root = grid->tree().root();
    for (u32 i = 0; i != nInt2s; ++i) {
      root.addChild(int2s[i]);
    }

    return OpenVDBStruct{grid};
  }

}  // namespace zs