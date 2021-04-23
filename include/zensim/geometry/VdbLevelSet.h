#pragma once
#include <any>
#include <string>

#include "zensim/container/DenseGrid.hpp"
#include "zensim/math/Vec.h"
#include "zensim/types/Tuple.h"

namespace zs {

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

  OpenVDBStruct particleArrayToGrid(const std::vector<std::array<float, 3>> &);
  bool writeGridToFile(const OpenVDBStruct &, std::string fn);

}  // namespace zs
