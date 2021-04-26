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

  OpenVDBStruct loadFloatGridFromVdbFile(const std::string &fn) {
    using GridType = openvdb::FloatGrid;
    using TreeType = GridType::TreeType;
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

  AdaptiveFloatGrid convertFloatGridToAdaptiveGrid(const OpenVDBStruct &grid,
                                                   const MemoryHandle mh) {
    using GridType = openvdb::FloatGrid;
    using TreeType = GridType::TreeType;
    using RootType = TreeType::RootNodeType;  // level 3 RootNode
    assert(RootType::LEVEL == 3);
    using Int1Type = RootType::ChildNodeType;  // level 2 InternalNode
    using Int2Type = Int1Type::ChildNodeType;  // level 1 InternalNode
    using LeafType = TreeType::LeafNodeType;   // level 0 LeafNode
    using SDFPtr = typename GridType::Ptr;
    const SDFPtr &gridPtr = grid.as<SDFPtr>();
    AdaptiveFloatGrid ret{mh};
    ret.dx = gridPtr->transform().voxelSize()[0];

    return ret;
  }

  void checkFloatGrid(OpenVDBStruct &grid) {
    using GridType = openvdb::FloatGrid;
    using TreeType = GridType::TreeType;
    using RootType = TreeType::RootNodeType;  // level 3 RootNode
    assert(RootType::LEVEL == 3);
    using Int1Type = RootType::ChildNodeType;  // level 2 InternalNode
    using Int2Type = Int1Type::ChildNodeType;  // level 1 InternalNode
    using LeafType = TreeType::LeafNodeType;   // level 0 LeafNode
    using SDFPtr = typename GridType::Ptr;
    SDFPtr &gridPtr = grid.as<SDFPtr>();
    fmt::print("grid meta -> voxel dx {}, {}, {}\n", gridPtr->transform().voxelSize()[0],
               gridPtr->transform().voxelSize()[1], gridPtr->transform().voxelSize()[2]);
    float dx = gridPtr->transform().voxelSize()[0];
    getchar();
#if 0
    for (GridType::ValueOnIter iter = gridPtr->beginValueOn(); iter.test(); ++iter) {
      // Print the coordinates of all voxels whose vector value has
      // a length greater than -10, and print the bounding box coordinates
      // of all tiles whose vector value length is greater than 10.
      // if (iter.getValue() > -10)
      if (iter.isValueOn()) {
        auto box = iter.getBoundingBox();
        auto st = box.getStart();
        auto ed = box.getEnd();

        fmt::print("iter-> level {}, box ({}, {}, {}) - ({}, {}, {}).\t", iter.getLevel(), st[0],
                   st[1], st[2], ed[0], ed[1], ed[2]);

        switch (iter.getDepth()) {
          case 0: {
            RootType *node = nullptr;
            iter.getNode<RootType>(node);
            if (node) {
              fmt::print("it is a root node");
            }
            break;
          }
          case 1: {
            Int1Type *node = nullptr;
            iter.getNode<Int1Type>(node);
            if (node) {
              fmt::print("it is a depth 1 node");
            }
            break;
          }
          case 2: {
            Int2Type *node = nullptr;
            iter.getNode<Int2Type>(node);
            if (node) {
              fmt::print("it is a depth 2 node");
            }
            break;
          }
          case 3: {
            LeafType *node = nullptr;
            iter.getNode<LeafType>(node);
            if (node) {
            }
            break;
          }
        }
        if (!iter.isValueOn()) {
          fmt::print("not active value???\n");
          getchar();
        }
        if (iter.isVoxelValue()) {
          fmt::print("voxel {}, {}, {}, value {}\n", iter.getCoord()[0], iter.getCoord()[1],
                     iter.getCoord()[2], iter.getValue());
        } else {
          fmt::print("wow, it is a box??? adaptive value {}\n", iter.getValue());
          getchar();
        }
      }
    }
#else
    for (TreeType::NodeIter iter = gridPtr->tree().beginNode(); iter.test(); ++iter) {
      // Print the coordinates of all voxels whose vector value has
      // a length greater than -10, and print the bounding box coordinates
      // of all tiles whose vector value length is greater than 10.
      // if (iter.getValue() > -10)
      auto box = iter.getBoundingBox();
      auto st = box.getStart();
      auto ed = box.getEnd();
      auto coord = iter.getCoord();

      fmt::print("node iter -> depth{}, level {}, box ({}, {}, {}) - ({}, {}, {}). ",
                 iter.getDepth(), iter.getLevel(), st[0], st[1], st[2], ed[0], ed[1], ed[2]);

      switch (iter.getDepth()) {
        case 0: {
          RootType *node = nullptr;
          iter.getNode<RootType>(node);
          fmt::print("root childCnt {}, tileCnt {}, voxelCnt {}\n", node->childCount(),
                     node->onTileCount(), node->onVoxelCount());
          if (node->onTileCount()) {
            fmt::print("has tile! ");
            getchar();
          }
          if (node) {
            // getchar();
            for (auto cell = node->beginValueOn(); cell; ++cell) {
              fmt::print("\troot local xyz child ({}, {}, {}) value {}\n", cell.getCoord()[0],
                         cell.getCoord()[1], cell.getCoord()[2], cell.getValue());
              getchar();
            }
          }
          break;
        }
        case 1: {
          Int1Type *node = nullptr;
          iter.getNode<Int1Type>(node);
          fmt::print("int1 childCnt {}, tileCnt {}, voxelCnt {}\n", node->childCount(),
                     node->onTileCount(), node->onVoxelCount());
          if (node->onTileCount()) {
            fmt::print("has tile! ");
            getchar();
          }
          if (node) {
            for (int i = 0; i < 32; ++i)
              for (int j = 0; j < 32; ++j)
                for (int k = 0; k < 32; ++k) {
                  if (node->isValueOn(openvdb::Coord(i, j, k)))
                    fmt::print("int1 local xyz child ({}, {}, {}) is active\n", i, j, k);
                }
            for (auto cell = node->beginValueOn(); cell; ++cell) {
              fmt::print("\tint1 local xyz child ({}, {}, {}) value {}\n", cell.getCoord()[0],
                         cell.getCoord()[1], cell.getCoord()[2], cell.getValue());
              getchar();
            }
          }
          break;
        }
        case 2: {
          Int2Type *node = nullptr;
          iter.getNode<Int2Type>(node);
          fmt::print("int2 childCnt {}, tileCnt {}, voxelCnt {}\n", node->childCount(),
                     node->onTileCount(), node->onVoxelCount());
          if (node->onTileCount()) {
            fmt::print("has tile! ");
            getchar();
          }
          if (node) {
            for (int i = 0; i < 16; ++i)
              for (int j = 0; j < 16; ++j)
                for (int k = 0; k < 16; ++k) {
                  if (node->isValueOn(openvdb::Coord(i, j, k)))
                    fmt::print("int2 local xyz child ({}, {}, {}) is active\n", i, j, k);
                }
            for (auto cell = node->beginValueOn(); cell; ++cell) {
              fmt::print("\tint2 local xyz child ({}, {}, {}) value {}\n", cell.getCoord()[0],
                         cell.getCoord()[1], cell.getCoord()[2], cell.getValue());
              getchar();
            }
          }
          break;
        }
        case 3: {
          LeafType *node = nullptr;
          iter.getNode<LeafType>(node);
          fmt::print("leaf childCnt {}, tileCnt {}, voxelCnt {}\n", node->childCount(),
                     node->onTileCount(), node->onVoxelCount());
          if (node->onTileCount()) {
            fmt::print("has tile! ");
            getchar();
          }
          if (node) {
            for (int i = 0; i < 8; ++i)
              for (int j = 0; j < 8; ++j)
                for (int k = 0; k < 8; ++k) {
                  if (node->isValueOn(openvdb::Coord(i, j, k)))
                    fmt::print("\tleaf local xyz child ({}, {}, {}) is active\n", i, j, k);
                }
            for (auto cell = node->beginValueOn(); cell; ++cell)
              fmt::print("\tleaf local xyz child ({}, {}, {}) value {}\n", cell.getCoord()[0],
                         cell.getCoord()[1], cell.getCoord()[2], cell.getValue());
            // getchar();
          }
          break;
        }
      }
    }
#endif
  }

  // https://www.openvdb.org/documentation/doxygen/codeExamples.html#sGettingMetadata
  tuple<DenseGrid<float, int, 3>, vec<float, 3>, vec<float, 3>> readPhiFromVdbFile(
      const std::string &fn, float dx) {
    constexpr int dim = 3;
    using TV = vec<float, dim>;
    using IV = vec<int, dim>;
    using GridT = typename openvdb::FloatGrid;
    using TreeT = typename GridT::TreeType;
    using VelGridT = typename openvdb::FloatGrid;
    using VelTreeT = typename VelGridT::TreeType;

    openvdb::io::File file(fn);
    file.open();
    openvdb::GridPtrVecPtr my_grids = file.getGrids();
    file.close();
    int count = 0;
    typename GridT::Ptr grid;
    for (openvdb::GridPtrVec::iterator iter = my_grids->begin(); iter != my_grids->end(); ++iter) {
      openvdb::GridBase::Ptr it = *iter;
      if ((*iter)->isType<GridT>()) {
        grid = openvdb::gridPtrCast<GridT>(*iter);
        count++;
        /// display meta data
        for (openvdb::MetaMap::MetaIterator it = grid->beginMeta(); it != grid->endMeta(); ++it) {
          const std::string &name = it->first;
          openvdb::Metadata::Ptr value = it->second;
          std::string valueAsString = value->str();
          std::cout << name << " = " << valueAsString << std::endl;
        }
      } else if ((*iter)->isType<VelGridT>()) {
        auto velgrid = openvdb::gridPtrCast<VelGridT>((*iter));
        for (openvdb::MetaMap::MetaIterator it = velgrid->beginMeta(); it != velgrid->endMeta();
             ++it) {
          const std::string &name = it->first;
          openvdb::Metadata::Ptr value = it->second;
          std::string valueAsString = value->str();
          std::cout << name << " = " << valueAsString << std::endl;
        }
      }
    }
    ZS_WARN_IF(count != 1, "Vdb file to load should only contain one levelset.");

    /// bounding box
    TV bmin, bmax;
    openvdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
    auto world_min = grid->indexToWorld(box.min());
    auto world_max = grid->indexToWorld(box.max());

    for (size_t d = 0; d < dim; d++) {
      bmin(d) = world_min[d];
      bmax(d) = world_max[d];
    }

    vec<int, 3> extents = ((bmax - bmin) / dx).cast<int>() + 1;

    /// phi
    auto sample = [&grid](const TV &X_input) -> float {
      TV X = X_input;
      openvdb::tools::GridSampler<TreeT, openvdb::tools::BoxSampler> interpolator(
          grid->constTree(), grid->transform());
      openvdb::math::Vec3<float> P(X(0), X(1), X(2));
      float phi = interpolator.wsSample(P);  // ws denotes world space
      return (float)phi;
    };
    printf(
        "Vdb file domain [%f, %f, %f] - [%f, %f, %f]; resolution {%d, %d, "
        "%d}\n",
        bmin(0), bmin(1), bmin(2), bmax(0), bmax(1), bmax(2), extents(0), extents(1), extents(2));
    // DenseGrid<float, int, 3> phi(extents, std::numeric_limits<float>::min());
    DenseGrid<float, int, 3> phi(extents, 2 * dx);
#pragma omp parallel for
    for (int x = 0; x < extents(0); ++x)
      for (int y = 0; y < extents(1); ++y)
        for (int z = 0; z < extents(2); ++z) {
          IV X = vec<int, 3>{x, y, z};
          TV x = X * dx + bmin;
          phi(X) = sample(x);
        }
    return zs::make_tuple(phi, bmin, bmax);
  }

}  // namespace zs