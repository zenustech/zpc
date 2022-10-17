#include <openvdb/Grid.h>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/VolumeToMesh.h>

#include "VdbLevelSet.h"
#include "zensim/Logger.hpp"
#include "zensim/execution/Concurrency.h"

namespace zs {

  OpenVDBStruct load_vec3fgrid_from_vdb_file(const std::string &fn) {
    using GridType = openvdb::Vec3fGrid;
    openvdb::io::File file(fn);
    file.open();
    openvdb::GridPtrVecPtr grids = file.getGrids();
    file.close();

    using SDFPtr = typename GridType::Ptr;
    SDFPtr grid;
    for (openvdb::GridPtrVec::iterator iter = grids->begin(); iter != grids->end(); ++iter) {
      openvdb::GridBase::Ptr it = *iter;
      if ((*iter)->isType<GridType>()) {
        grid = openvdb::gridPtrCast<GridType>(*iter);
        /// display meta data
        for (openvdb::MetaMap::MetaIterator it = grid->beginMeta(); it != grid->endMeta(); ++it) {
          const std::string &name = it->first;
          openvdb::Metadata::Ptr value = it->second;
          std::string valueAsString = value->str();
          std::cout << name << " = " << valueAsString << std::endl;
        }
        break;
      }
    }
    return OpenVDBStruct{grid};
  }
  SparseLevelSet<3> convert_vdblevelset_to_sparse_levelset(const OpenVDBStruct &sdf,
                                                           const OpenVDBStruct &vel) {
    using SDFGridType = openvdb::FloatGrid;
    using SDFTreeType = SDFGridType::TreeType;
    using SDFRootType = SDFTreeType::RootNodeType;  // level 3 RootNode
    assert(SDFRootType::LEVEL == 3);
    using SDFInt1Type = SDFRootType::ChildNodeType;  // level 2 InternalNode
    using SDFInt2Type = SDFInt1Type::ChildNodeType;  // level 1 InternalNode
    using SDFLeafType = SDFTreeType::LeafNodeType;   // level 0 LeafNode

    using VelGridType = openvdb::Vec3fGrid;
    using VelTreeType = VelGridType::TreeType;
    using VelRootType = VelTreeType::RootNodeType;  // level 3 RootNode
    assert(VelRootType::LEVEL == 3);
    using VelInt1Type = VelRootType::ChildNodeType;  // level 2 InternalNode
    using VelInt2Type = VelInt1Type::ChildNodeType;  // level 1 InternalNode
    using VelLeafType = VelTreeType::LeafNodeType;   // level 0 LeafNode

    using SDFPtr = typename SDFGridType::Ptr;
    using VelPtr = typename VelGridType::Ptr;
    const SDFPtr &sdfGridPtr = sdf.as<SDFPtr>();
    const VelPtr &velGridPtr = vel.as<VelPtr>();
    auto velAccessor = velGridPtr->getConstAccessor();
    openvdb::tools::GridSampler<VelGridType::ConstAccessor, openvdb::tools::BoxSampler> velSampler{
        velAccessor, velGridPtr->transform()};
    auto sdfTransform = sdfGridPtr->transform();

    using SpLs = SparseLevelSet<3, grid_e::collocated>;

    using IV = typename SpLs::IV;
    using TV = vec<typename SpLs::value_type, 3>;

    sdfGridPtr->tree().voxelizeActiveTiles();
    velGridPtr->tree().voxelizeActiveTiles();
    SpLs ret{};
    const auto leafCount = sdfGridPtr->tree().leafCount();
    ret._backgroundValue = sdfGridPtr->background();
    {
      auto v = velGridPtr->background();
      ret._backgroundVecValue = TV{v[0], v[1], v[2]};
    }
    ret._table = typename SparseLevelSet<3>::table_t{leafCount, memsrc_e::host, -1};
    ret._grid = typename SpLs::grid_t{{{"sdf", 1}, {"v", 3}},
                                      (float)sdfGridPtr->transform().voxelSize()[0],
                                      leafCount,
                                      memsrc_e::host,
                                      -1};
    {
      openvdb::CoordBBox box = sdfGridPtr->evalActiveVoxelBoundingBox();
      auto mi = box.min();
      auto ma = box.max();
      for (int d = 0; d != 3; ++d) {
        ret._min[d] = mi[d];
        ret._max[d] = ma[d];
      }
    }
    openvdb::Mat4R v2w = sdfGridPtr->transform().baseMap()->getAffineMap()->getMat4();

    sdfGridPtr->transform().print();

    vec<float, 4, 4> lsv2w;
    for (auto &&[r, c] : ndrange<2>(4)) lsv2w(r, c) = v2w[r][c];
    ret.resetTransformation(lsv2w);

    {
      auto [mi, ma] = proxy<execspace_e::host>(ret).getBoundingBox();
      fmt::print(
          "leaf count: {}. background value: {}. dx: {}. ibox: [{}, {}, {} ~ {}, {}, {}], wbox: "
          "[{}, {}, {} ~ {}, {}, {}]\n",
          leafCount, ret._backgroundValue, ret._grid.dx, ret._min[0], ret._min[1], ret._min[2],
          ret._max[0], ret._max[1], ret._max[2], mi[0], mi[1], mi[2], ma[0], ma[1], ma[2]);
    }

    auto table = proxy<execspace_e::host>(ret._table);
    // auto tiles = proxy<execspace_e::host>({"sdf", "v"}, ret._tiles);
    auto gridview = proxy<execspace_e::host>(ret._grid);
    ret._table.reset(true);

    SDFTreeType::LeafCIter sdfIter = sdfGridPtr->tree().cbeginLeaf();
    // VelTreeType::LeafCIter velIter = velGridPtr->tree().cbeginLeaf();
    for (; sdfIter; ++sdfIter) {
      const SDFTreeType::LeafNodeType &sdfNode = *sdfIter;
      // const VelTreeType::LeafNodeType &velNode = *velIter;
      // if (sdfNode.onVoxelCount() != velNode.onVoxelCount()) {
      //   fmt::print("sdf grid and vel grid structure not consistent!\n");
      // }
      if (sdfNode.onVoxelCount() > 0) {
        IV coord{};
        {
          auto cell = sdfNode.beginValueOn();
          for (int d = 0; d < SparseLevelSet<3>::table_t::dim; ++d) coord[d] = cell.getCoord()[d];
        }

        auto blockid = coord;
        for (int d = 0; d < SparseLevelSet<3>::table_t::dim; ++d)
          blockid[d] -= (coord[d] & (ret.side_length - 1));
        auto blockno = table.insert(blockid);

        RM_CVREF_T(blockno) cellid = 0;
        auto sdfCell = sdfNode.beginValueAll();
        // auto velCell = velNode.beginValueAll();
        for (; sdfCell; ++sdfCell, ++cellid) {
          auto sdf = sdfCell.getValue();
          // auto vel = velCell.getValue();
          auto vel = velSampler.wsSample(sdfTransform.indexToWorld(sdfCell.getCoord()));
          const auto offset = blockno * ret.block_size + cellid;
          gridview.voxel("sdf", offset) = sdf;
          gridview.set("v", offset, TV{vel[0], vel[1], vel[2]});
        }
      }
    }
    return ret;
  }
  SparseLevelSet<3> convert_vdblevelset_to_sparse_levelset(const OpenVDBStruct &sdf,
                                                           const OpenVDBStruct &vel,
                                                           const MemoryHandle mh) {
    return convert_vdblevelset_to_sparse_levelset(sdf, vel).clone(mh);
  }

}  // namespace zs