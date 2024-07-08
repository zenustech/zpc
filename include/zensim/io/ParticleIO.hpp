#pragma once
#include <array>
#include <string>
#include <vector>

#include "zensim/math/Vec.h"

namespace zs {

  template <typename T, size_t dim>
  void write_partio(std::string filename, const std::vector<std::array<T, dim>> &data,
                    std::string tag = std::string{"position"});

  template <typename T, size_t dim>
  void write_partio_with_stress(std::string filename, const std::vector<std::array<T, dim>> &data,
                                const std::vector<T> &stressData);

  template <typename T, size_t dim>
  void write_partio_with_grid(std::string filename, const std::vector<std::array<T, dim>> &pos,
                              const std::vector<std::array<T, dim>> &force);

  extern template ZPC_BACKEND_TEMPLATE_IMPORT void write_partio<float, 3>(std::string, const std::vector<std::array<float, 3>> &,
                                       std::string);
  extern template ZPC_BACKEND_TEMPLATE_IMPORT void write_partio<double, 3>(std::string, const std::vector<std::array<double, 3>> &,
                                        std::string);

  extern template ZPC_BACKEND_TEMPLATE_IMPORT void write_partio_with_stress(std::string, const std::vector<std::array<float, 3>> &,
                                         const std::vector<float> &);
  extern template ZPC_BACKEND_TEMPLATE_IMPORT void write_partio_with_stress(std::string, const std::vector<std::array<double, 3>> &,
                                         const std::vector<double> &);

  extern template ZPC_BACKEND_TEMPLATE_IMPORT void write_partio_with_grid(std::string, const std::vector<std::array<float, 3>> &,
                                       const std::vector<std::array<float, 3>> &);
  extern template ZPC_BACKEND_TEMPLATE_IMPORT void write_partio_with_grid(std::string, const std::vector<std::array<double, 3>> &,
                                       const std::vector<std::array<double, 3>> &);

}  // namespace zs
