#pragma once
#include <any>
#include <string>

#include "zensim/container/DenseGrid.hpp"
#include "zensim/geometry/SparseLevelSet.hpp"
#include "zensim/math/Vec.h"
#include "zensim/types/Tuple.h"

namespace zs {

  void initialize_openvdb();

  struct OpenVDBStruct {
    OpenVDBStruct() noexcept = default;
    template <typename T> constexpr OpenVDBStruct(T &&obj) : object{FWD(obj)} {}
    template <typename T> T &as() { return std::any_cast<T &>(object); }
    template <typename T> const T &as() const { return std::any_cast<const T &>(object); }
    template <typename T> bool is() const noexcept { return object.type() == typeid(T); }

    std::any object;
  };

  OpenVDBStruct load_floatgrid_from_mesh_file(const std::string &fn, float h);
  OpenVDBStruct load_floatgrid_from_vdb_file(const std::string &fn);
  OpenVDBStruct load_vec3fgrid_from_vdb_file(const std::string &fn);
  bool write_floatgrid_to_vdb_file(std::string_view fn, const OpenVDBStruct &grid);

  ///
  std::vector<const void *> get_floatgrid_interior_leaves(const OpenVDBStruct &grid);

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
  SparseLevelSet<3> convert_vdblevelset_to_sparse_levelset(const OpenVDBStruct &sdf,
                                                           const OpenVDBStruct &vel);
  SparseLevelSet<3> convert_vdblevelset_to_sparse_levelset(const OpenVDBStruct &sdf,
                                                           const OpenVDBStruct &vel,
                                                           const MemoryHandle mh);

  void check_floatgrid(OpenVDBStruct &grid);
  OpenVDBStruct particlearray_to_pointdatagrid(const std::vector<std::array<float, 3>> &);
  std::vector<std::array<float, 3>> pointdatagrid_to_particlearray(const OpenVDBStruct &);
  bool write_pointdatagrid_to_file(const OpenVDBStruct &, std::string fn);

}  // namespace zs
