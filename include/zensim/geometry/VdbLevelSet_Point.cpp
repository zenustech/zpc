#include <openvdb/Grid.h>
#include <openvdb/io/File.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointCount.h>

#include "VdbLevelSet.h"
#include "zensim/types/Iterator.h"

namespace zs {

  OpenVDBStruct particleArrayToGrid(const std::vector<std::array<float, 3>> &particles) {
    std::vector<openvdb::Vec3R> positions(particles.size());
    for (auto &&[dst, src] : zip(positions, particles))
      for (int d = 0; d < 3; ++d) dst[d] = src[d];
    // The VDB Point-Partioner is used when bucketing points and requires a
    // specific interface. For convenience, we use the PointAttributeVector
    // wrapper around an stl vector wrapper here, however it is also possible to
    // write one for a custom data structure in order to match the interface
    // required.
    openvdb::points::PointAttributeVector<openvdb::Vec3R> positionsWrapper(positions);
    // This method computes a voxel-size to match the number of
    // points / voxel requested. Although it won't be exact, it typically offers
    // a good balance of memory against performance.
    int pointsPerVoxel = 8;
    float voxelSize = openvdb::points::computeVoxelSize(positionsWrapper, pointsPerVoxel);
    // Print the voxel-size to cout
    std::cout << "VoxelSize=" << voxelSize << std::endl;
    // Create a transform using this voxel-size.
    openvdb::math::Transform::Ptr transform
        = openvdb::math::Transform::createLinearTransform(voxelSize);

    // Create a PointDataGrid containing these four points and using the
    // transform given. This function has two template parameters, (1) the codec
    // to use for storing the position, (2) the grid we want to create
    // (ie a PointDataGrid).
    // We use no compression here for the positions.
    using PDGPtr = openvdb::points::PointDataGrid::Ptr;
    OpenVDBStruct ret = openvdb::points::createPointDataGrid<openvdb::points::NullCodec,
                                                             openvdb::points::PointDataGrid>(
        positions, *transform);
    ret.as<PDGPtr>()->setName("Points");
    return ret;
  }

  bool writeGridToFile(const OpenVDBStruct &grid, std::string fn) {
    using PDGPtr = openvdb::points::PointDataGrid::Ptr;
    if (!grid.is<PDGPtr>()) return false;
    // Create a VDB file object and write out the grid.
    openvdb::io::File(fn).write({grid.as<PDGPtr>()});
    return true;
  }

}  // namespace zs