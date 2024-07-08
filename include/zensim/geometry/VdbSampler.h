#pragma once
#include <array>
#include <string>
#include <vector>

#include "VdbLevelSet.h"

namespace zs {

  ZPC_EXTENSION_API std::vector<std::array<float, 3>> sample_from_floatgrid(
      const OpenVDBStruct &grid, float dx, float ppc);
  ZPC_EXTENSION_API std::vector<std::array<float, 3>> sample_from_vdb_file(
      const std::string &filename, float dx, float ppc);
  ZPC_EXTENSION_API std::vector<std::array<float, 3>> sample_from_obj_file(
      const std::string &filename, float dx, float ppc);
  ZPC_EXTENSION_API std::vector<std::array<float, 3>> sample_from_levelset(const OpenVDBStruct &ls,
                                                                           float dx, float ppc);

}  // namespace zs
