#pragma once
#include <any>
#include <string>

#include "zensim/container/DenseGrid.hpp"
#include "zensim/geometry/AdaptiveGrid.hpp"
#include "zensim/geometry/SparseGrid.hpp"
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

  ZPC_EXTENSION_API OpenVDBStruct load_floatgrid_from_mesh_file(const std::string &fn, float h);
  ZPC_EXTENSION_API OpenVDBStruct load_floatgrid_from_vdb_file(const std::string &fn);
  ZPC_EXTENSION_API OpenVDBStruct load_vec3fgrid_from_vdb_file(const std::string &fn);
  ZPC_EXTENSION_API bool write_floatgrid_to_vdb_file(std::string_view fn, const OpenVDBStruct &grid);

  /// adaptive grid
  ZPC_EXTENSION_API VdbGrid<3, f32, index_sequence<3, 4, 5>> convert_floatgrid_to_adaptive_grid(
      const OpenVDBStruct &grid, SmallString propTag = "sdf");
  ZPC_EXTENSION_API VdbGrid<3, f32, index_sequence<3, 4, 5>> convert_floatgrid_to_adaptive_grid(
      const OpenVDBStruct &grid, const MemoryHandle mh, SmallString propTag = "sdf");
  ZPC_EXTENSION_API OpenVDBStruct convert_adaptive_grid_to_floatgrid(
      const VdbGrid<3, f32, index_sequence<3, 4, 5>> &agIn, SmallString propTag = "sdf",
      u32 gridClass = 1u, SmallString gridName = "sdf");

  ZPC_EXTENSION_API VdbGrid<3, f32, index_sequence<3, 4, 5>> convert_float3grid_to_adaptive_grid(
      const OpenVDBStruct &grid, SmallString propTag = "v");
  ZPC_EXTENSION_API VdbGrid<3, f32, index_sequence<3, 4, 5>> convert_float3grid_to_adaptive_grid(
      const OpenVDBStruct &grid, const MemoryHandle mh, SmallString propTag = "v");
  ZPC_EXTENSION_API OpenVDBStruct convert_adaptive_grid_to_float3grid(
      const VdbGrid<3, f32, index_sequence<3, 4, 5>> &grid, SmallString propTag = "v",
      SmallString gridName = "SparseGrid");

  ZPC_EXTENSION_API void assign_floatgrid_to_adaptive_grid(const OpenVDBStruct &grid,
                                         VdbGrid<3, f32, index_sequence<3, 4, 5>> &ag_,
                                         SmallString propTag);
  ZPC_EXTENSION_API void assign_float3grid_to_adaptive_grid(const OpenVDBStruct &grid,
                                          VdbGrid<3, f32, index_sequence<3, 4, 5>> &ag_,
                                          SmallString propTag);

  /// floatgrid
  template <typename SplsT> OpenVDBStruct convert_sparse_levelset_to_vdbgrid(const SplsT &grid);
  ZPC_EXTENSION_API SparseLevelSet<3> convert_floatgrid_to_sparse_levelset(const OpenVDBStruct &grid);
  ZPC_EXTENSION_API SparseLevelSet<3> convert_floatgrid_to_sparse_levelset(const OpenVDBStruct &grid,
                                                                 const MemoryHandle mh);

  ZPC_EXTENSION_API OpenVDBStruct convert_sparse_grid_to_floatgrid(const SparseGrid<3, f32, 8> &grid,
                                                         SmallString propTag = "sdf",
                                                         u32 gridClass = 1,
                                                         SmallString gridName = "SparseGrid");
  ZPC_EXTENSION_API void assign_floatgrid_to_sparse_grid(const OpenVDBStruct &grid,
                                               SparseGrid<3, f32, 8> &spg,
                                               SmallString propTag = "sdf");
  ZPC_EXTENSION_API void assign_float3grid_to_sparse_grid(const OpenVDBStruct &grid,
                                                SparseGrid<3, f32, 8> &spg,
                                                SmallString propTag = "v");
  ZPC_EXTENSION_API SparseGrid<3, f32, 8> convert_floatgrid_to_sparse_grid(const OpenVDBStruct &grid,
                                                                 SmallString propTag = "sdf");
  ZPC_EXTENSION_API SparseGrid<3, f32, 8> convert_floatgrid_to_sparse_grid(const OpenVDBStruct &grid,
                                                                 const MemoryHandle mh,
                                                                 SmallString propTag = "sdf");
  // Staggered only
  ZPC_EXTENSION_API OpenVDBStruct convert_sparse_grid_to_float3grid(const SparseGrid<3, f32, 8> &grid,
                                                          SmallString propTag = "v",
                                                          SmallString gridName = "SparseGrid");
  ZPC_EXTENSION_API SparseGrid<3, f32, 8> convert_float3grid_to_sparse_grid(const OpenVDBStruct &grid,
                                                                  SmallString propTag = "v");
  ZPC_EXTENSION_API SparseGrid<3, f32, 8> convert_float3grid_to_sparse_grid(const OpenVDBStruct &grid,
                                                                  const MemoryHandle mh,
                                                                  SmallString propTag = "v");

  /// float3grid
  ZPC_EXTENSION_API SparseLevelSet<3> convert_vec3fgrid_to_sparse_levelset(const OpenVDBStruct &grid);
  ZPC_EXTENSION_API SparseLevelSet<3> convert_vec3fgrid_to_sparse_levelset(const OpenVDBStruct &grid,
                                                                 const MemoryHandle mh);

  ZPC_EXTENSION_API SparseLevelSet<3, grid_e::staggered> convert_vec3fgrid_to_sparse_staggered_grid(
      const OpenVDBStruct &grid);
  ZPC_EXTENSION_API SparseLevelSet<3, grid_e::staggered> convert_vec3fgrid_to_sparse_staggered_grid(
      const OpenVDBStruct &grid, const MemoryHandle mh);

  /// floatgrid + float3grid
  ZPC_EXTENSION_API SparseLevelSet<3> convert_vdblevelset_to_sparse_levelset(const OpenVDBStruct &sdf,
                                                                   const OpenVDBStruct &vel);
  ZPC_EXTENSION_API SparseLevelSet<3> convert_vdblevelset_to_sparse_levelset(const OpenVDBStruct &sdf,
                                                                   const OpenVDBStruct &vel,
                                                                   const MemoryHandle mh);

  void check_floatgrid(OpenVDBStruct &grid);
  OpenVDBStruct particlearray_to_pointdatagrid(const std::vector<std::array<float, 3>> &);
  std::vector<std::array<float, 3>> pointdatagrid_to_particlearray(const OpenVDBStruct &);
  bool write_pointdatagrid_to_file(const OpenVDBStruct &, std::string fn);

}  // namespace zs
