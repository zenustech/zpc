#include <openvdb/Grid.h>
#include <openvdb/io/File.h>
#include <openvdb/math/Transform.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h>

#include <fstream>

#include "VdbLevelSet.h"
#include "zensim/Logger.hpp"
#include "zensim/execution/Concurrency.h"
#include "zensim/memory/MemoryResource.h"
#include "zensim/types/Property.h"

namespace zs {

  void initialize_openvdb() { openvdb::initialize(); }

  OpenVDBStruct load_floatgrid_from_mesh_file(const std::string &filename, float h) {
    using Vec3ui = openvdb::math::Vec3ui;
    std::vector<openvdb::Vec3f> vertList;
    std::vector<Vec3ui> faceList;
    std::ifstream infile(filename);
    if (!infile) {
      std::cerr << "Failed to open. Terminating.\n";
      exit(-1);
    }

    int ignored_lines = 0;
    std::string line;

    while (!infile.eof()) {
      std::getline(infile, line);
      auto ed = line.find_first_of(" ");
      if (line.substr(0, ed) == std::string("v")) {
        std::stringstream data(line);
        char c;
        openvdb::Vec3f point;
        data >> c >> point[0] >> point[1] >> point[2];
        vertList.push_back(point);
      } else if (line.substr(0, ed) == std::string("f")) {
        std::stringstream data(line);
        char c;
        int v0, v1, v2;
        data >> c >> v0 >> v1 >> v2;
        faceList.push_back(Vec3ui(v0 - 1, v1 - 1, v2 - 1));
      } else {
        ++ignored_lines;
      }
    }
    infile.close();
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    points.resize(vertList.size());
    triangles.resize(faceList.size());
#if ZS_ENABLE_OPENMP
#  pragma omp parallel for
#endif
    for (int p = 0; p < vertList.size(); ++p)
      points[p] = openvdb::Vec3s(vertList[p][0], vertList[p][1], vertList[p][2]);
      // tbb::parallel_for(0, (int)vertList.size(), 1, [&](int p) {
      //  points[p] = openvdb::Vec3s(vertList[p][0], vertList[p][1], vertList[p][2]);
      //});
#if ZS_ENABLE_OPENMP
#  pragma omp parallel for
#endif
    for (int p = 0; p < faceList.size(); ++p)
      triangles[p] = openvdb::Vec3I(faceList[p][0], faceList[p][1], faceList[p][2]);
    // tbb::parallel_for(0, (int)faceList.size(), 1, [&](int p) {
    //  triangles[p] = openvdb::Vec3I(faceList[p][0], faceList[p][1], faceList[p][2]);
    //});
    openvdb::FloatGrid::Ptr grid = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(
        *openvdb::math::Transform::createLinearTransform(h), points, triangles, 3.0);
    return OpenVDBStruct{grid};
  }

  OpenVDBStruct load_floatgrid_from_vdb_file(const std::string &fn) {
    using GridType = openvdb::FloatGrid;
    openvdb::io::File file(fn);
    file.open();
    openvdb::GridPtrVecPtr grids = file.getGrids();
    file.close();

    fmt::print("{} grid{} in file \"{}\"\n", grids->size(), grids->size() > 1 ? "s" : "", fn);
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

  bool write_floatgrid_to_vdb_file(std::string_view fn, const OpenVDBStruct &grid) {
    openvdb::io::File file(fn.data());
    // if (!file.isOpen()) return false;
    openvdb::GridPtrVec grids{};
    const auto &gridPtr = grid.as<typename openvdb::FloatGrid::Ptr>();
    grids.push_back(gridPtr);
    file.write(grids);
    file.close();
    return true;
  }

  void check_floatgrid(OpenVDBStruct &grid) {
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
    i64 leafCount = 0;
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
          leafCount++;
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
    fmt::print("check grid: leaf count {}\n", leafCount);
#endif
  }

}  // namespace zs