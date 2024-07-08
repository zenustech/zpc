#include "ParticleIO.hpp"

#include "zensim/zpc_tpls/partio/Partio.h"

namespace zs {

  template <typename T, size_t dim>
  void write_partio(std::string filename, const std::vector<std::array<T, dim>> &data,
                    std::string tag) {
    Partio::ParticlesDataMutable *parts = Partio::create();

    Partio::ParticleAttribute attrib = parts->addAttribute(tag.c_str(), Partio::VECTOR, 3);

    parts->addParticles(data.size());
    for (int idx = 0; idx < (int)data.size(); ++idx) {
      float *val = parts->dataWrite<float>(attrib, idx);
      val[0] = data[idx][0];
      if constexpr (dim > 1) {
        val[1] = data[idx][1];
        if constexpr (dim > 2)
          val[2] = data[idx][2];
        else
          val[2] = 0.f;
      } else
        val[1] = val[2] = 0.f;
    }
    Partio::write(filename.c_str(), *parts);
    parts->release();
  }

  template <typename T, size_t dim>
  void write_partio_with_stress(std::string filename, const std::vector<std::array<T, dim>> &data,
                                const std::vector<T> &stressData) {
    Partio::ParticlesDataMutable *parts = Partio::create();

    Partio::ParticleAttribute pattrib = parts->addAttribute("position", Partio::VECTOR, 3);
    Partio::ParticleAttribute sattrib = parts->addAttribute("stress", Partio::FLOAT, 1);

    parts->addParticles(data.size());
    for (int idx = 0; idx < (int)data.size(); ++idx) {
      float *val = parts->dataWrite<float>(pattrib, idx);
      float *stress = parts->dataWrite<float>(sattrib, idx);
      val[0] = data[idx][0];
      if constexpr (dim > 1) {
        val[1] = data[idx][1];
        if constexpr (dim > 2)
          val[2] = data[idx][2];
        else
          val[2] = 0.f;
      } else
        val[1] = val[2] = 0.f;
      stress[0] = stressData[idx];
    }
    Partio::write(filename.c_str(), *parts);
    parts->release();
  }

  template <typename T, size_t dim>
  void write_partio_with_grid(std::string filename, const std::vector<std::array<T, dim>> &pos,
                              const std::vector<std::array<T, dim>> &force) {
    Partio::ParticlesDataMutable *parts = Partio::create();

    Partio::ParticleAttribute pattrib = parts->addAttribute("position", Partio::VECTOR, dim);
    Partio::ParticleAttribute fattrib = parts->addAttribute("force", Partio::VECTOR, dim);

    parts->addParticles(pos.size());
    for (int idx = 0; idx < (int)pos.size(); ++idx) {
      float *p = parts->dataWrite<float>(pattrib, idx);
      float *f = parts->dataWrite<float>(fattrib, idx);
      for (int k = 0; k < dim; ++k) {
        p[k] = pos[idx][k];
        f[k] = force[idx][k];
      }
    }
    Partio::write(filename.c_str(), *parts);
    parts->release();
  }

  template ZPC_BACKEND_TEMPLATE_EXPORT void write_partio<float, 3>(std::string, const std::vector<std::array<float, 3>> &,
                                       std::string);
  template ZPC_BACKEND_TEMPLATE_EXPORT void write_partio<double, 3>(std::string, const std::vector<std::array<double, 3>> &,
                                        std::string);

  template ZPC_BACKEND_TEMPLATE_EXPORT void write_partio_with_stress(std::string, const std::vector<std::array<float, 3>> &,
                                         const std::vector<float> &);
  template ZPC_BACKEND_TEMPLATE_EXPORT void write_partio_with_stress(std::string, const std::vector<std::array<double, 3>> &,
                                         const std::vector<double> &);

  template ZPC_BACKEND_TEMPLATE_EXPORT void write_partio_with_grid(std::string, const std::vector<std::array<float, 3>> &,
                                       const std::vector<std::array<float, 3>> &);
  template ZPC_BACKEND_TEMPLATE_EXPORT void write_partio_with_grid(std::string, const std::vector<std::array<double, 3>> &,
                                       const std::vector<std::array<double, 3>> &);

}  // namespace zs