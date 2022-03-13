#pragma once
#include <array>
#include <string>
#include <vector>

#include "zensim/math/Vec.h"

namespace zs {

  template <typename T, std::size_t dim>
  void write_partio(std::string filename, const std::vector<std::array<T, dim>> &data,
                    std::string tag = std::string{"position"});

  template <typename T, std::size_t dim>
  void write_partio_with_stress(std::string filename, const std::vector<std::array<T, dim>> &data,
                                const std::vector<T> &stressData);

  template <typename T, std::size_t dim>
  void write_partio_with_grid(std::string filename, const std::vector<std::array<T, dim>> &pos,
                              const std::vector<std::array<T, dim>> &force);

}  // namespace zs
