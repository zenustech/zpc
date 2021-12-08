#pragma once
#include <any>
#include <string>

#include "zensim/container/DenseGrid.hpp"
#include "zensim/geometry/SparseLevelSet.hpp"
#include "zensim/math/Vec.h"
#include "zensim/types/Tuple.h"

namespace zs {

  void initialize_openvdb();

  tuple<DenseGrid<float, int, 3>, vec<float, 3>, vec<float, 3>> readPhiFromVdbFile(
      const std::string &fn, float dx);

  tuple<DenseGrid<float, int, 3>, DenseGrid<vec<float, 3>, int, 3>, vec<float, 3>, vec<float, 3>>
  readPhiVelFromVdbFile(const std::string &fn, float dx);

  struct OpenVDBStruct {
    template <typename T> constexpr OpenVDBStruct(T &&obj) : object{FWD(obj)} {}
    template <typename T> T &as() { return std::any_cast<T &>(object); }
    template <typename T> const T &as() const { return std::any_cast<const T &>(object); }
    template <typename T> bool is() const noexcept { return object.type() == typeid(T); }

    std::any object;
  };

  OpenVDBStruct load_floatgrid_from_vdb_file(const std::string &fn);
  OpenVDBStruct load_vec3fgrid_from_vdb_file(const std::string &fn);
  bool writeFloatGridToVdbFile(std::string_view fn, const OpenVDBStruct &grid);

  /// floatgrid
  OpenVDBStruct convert_sparse_levelset_to_floatgrid(const SparseLevelSet<3> &grid);

  SparseLevelSet<3> convert_floatgrid_to_sparse_levelset(const OpenVDBStruct &grid);
  SparseLevelSet<3> convert_floatgrid_to_sparse_levelset(const OpenVDBStruct &grid,
                                                         const MemoryHandle mh);

  /// float3grid
  SparseLevelSet<3> convert_vec3fgrid_to_sparse_levelset(const OpenVDBStruct &grid);
  SparseLevelSet<3> convert_vec3fgrid_to_sparse_levelset(const OpenVDBStruct &grid,
                                                         const MemoryHandle mh);

  /// floatgrid + float3grid
  SparseLevelSet<3> convertLevelSetGridToSparseLevelSet(const OpenVDBStruct &sdf,
                                                        const OpenVDBStruct &vel);
  SparseLevelSet<3> convertLevelSetGridToSparseLevelSet(const OpenVDBStruct &sdf,
                                                        const OpenVDBStruct &vel,
                                                        const MemoryHandle mh);

  void checkFloatGrid(OpenVDBStruct &grid);
  OpenVDBStruct particleArrayToGrid(const std::vector<std::array<float, 3>> &);
  std::vector<std::array<float, 3>> particleGridToArray(const OpenVDBStruct &);
  bool writeGridToFile(const OpenVDBStruct &, std::string fn);

}  // namespace zs
