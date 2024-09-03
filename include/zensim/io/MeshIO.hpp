#pragma once
#include <array>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "zensim/geometry/Mesh.hpp"
#include "zensim/math/Vec.h"
#include "zensim/types/Optional.h"

namespace zs {

  ZPC_API bool load_obj(std::string_view file, std::vector<std::array<float, 3>> *pos,
                        std::vector<std::array<float, 3>> *nrm,
                        std::vector<std::array<float, 2>> *uv,
                        std::vector<std::array<u32, 3>> *tris);

  template <typename T, int dim, typename Tn>
  bool read_tri_mesh_obj(const std::string &file, Mesh<T, dim, Tn, 3> &mesh) {
    auto nV = mesh.nodes.size();
    auto nE = mesh.elems.size();
    bool ret = load_obj(file, &mesh.nodes, &mesh.norms, &mesh.uvs, &mesh.elems);
    printf("mesh append: pos, tri [%d, %d] -> [%d, %d]\n", nV, nE, mesh.nodes.size(),
           mesh.elems.size());
    return ret;
  }

  template <typename T, typename Tn>
  bool read_tet_mesh_vtk(const std::string &file, Mesh<T, 3, Tn, 4> &mesh) {
    // TetMesh
    std::ifstream in(file);
    if (!in || file.empty()) {
      printf("%s not found!\n", file.c_str());
      return false;
    }

    using Node = typename Mesh<T, 3, Tn, 4>::Node;
    using Elem = typename Mesh<T, 3, Tn, 4>::Elem;
    auto &X = mesh.nodes;
    auto &indices = mesh.elems;

    Tn initial_X_size = X.size();
    Tn initial_indices_size = indices.size();

    std::string line;
    Node position;

    bool reading_points = false;
    bool reading_tets = false;
    size_t n_points = 0;
    size_t n_tets = 0;

    while (std::getline(in, line)) {
      std::stringstream ss(line);
      if (line.size() == (size_t)(0)) {
      } else if (line.substr(0, 6) == "POINTS") {
        reading_points = true;
        reading_tets = false;
        ss.ignore(128, ' ');  // ignore "POINTS"
        ss >> n_points;
      } else if (line.substr(0, 5) == "CELLS") {
        reading_points = false;
        reading_tets = true;
        ss.ignore(128, ' ');  // ignore "CELLS"
        ss >> n_tets;
      } else if (line.substr(0, 10) == "CELL_TYPES") {
        reading_points = false;
        reading_tets = false;
      } else if (reading_points) {
        for (int i = 0; i < 3; i++) ss >> position[i];
        X.emplace_back(position);
      } else if (reading_tets) {
        ss.ignore(128, ' ');  // ignore "4"
        Elem tet;
        for (int i = 0; i < 4; i++) {
          ss >> tet[i];
          tet[i] += initial_X_size;
        }
        indices.emplace_back(tet);
      }
    }
    in.close();

    assert((n_points == X.size() - initial_X_size) && "vtk read X count doesn't match.");
    assert((n_tets == indices.size() - initial_indices_size)
           && "vtk read element count doesn't match.");
    printf("positions, tetrahedra [%d, %d] -> [%d, %d]\n", initial_X_size, initial_indices_size,
           (int)X.size(), (int)indices.size());
    return true;
  }

  template <typename T, int dim, typename Tn>
  bool write_tri_mesh_obj(const std::string &filename, const Mesh<T, dim, Tn, 3> &mesh) {
    std::ofstream out(filename.c_str(), std::ios::out);
    if (!out) {
      printf("failed to create file %s\n", filename.c_str());
      return false;
    }
    const auto &nodes = mesh.nodes;
    const auto &faces = mesh.elems;
    for (const auto &node : nodes) {
      out << "v";
      for (int d = 0; d < dim; ++d) out << " " << node[d];
      if constexpr (dim == 2)
        out << " 0\n";
      else if constexpr (dim == 3)
        out << '\n';
    }
    out << "\n";
    for (const auto &face : faces)
      out << "f " << face[0] + 1 << " " << face[1] + 1 << " " << face[2] + 1 << "\n";

    out.close();
    printf("done writing to %s\n", filename.c_str());
    return true;
  }

  template <typename T, typename Tn>
  bool write_tet_mesh_vtk(const std::string &filename, const Mesh<T, 3, Tn, 4> &mesh) {
    assert((mesh.nodes.size() != (size_t)0) && "The X array for writing tetmesh vtk is empty.");
    assert((mesh.elems.size() != (size_t)0)
           && "The tet mesh data structure for writing tetmesh vtk is empty.");

    std::ofstream os(filename);
    if (!os) {
      printf("failed to create file %s\n", filename.c_str());
      return false;
    }

    using Node = typename Mesh<T, 3, Tn, 4>::Node;
    using Elem = typename Mesh<T, 3, Tn, 4>::Elem;
    auto &X = mesh.nodes;
    auto &indices = mesh.elems;

    os << "# vtk DataFile Version 2.0\n";
    os << "Unstructured Grid\n";
    os << "ASCII\n";
    os << "DATASET UNSTRUCTURED_GRID\n";

    os << "POINTS " << X.size() << " ";
    if (std::is_same<T, float>::value)
      os << "float\n";
    else
      os << "double\n";

    for (const auto &x : X) os << x[0] << " " << x[1] << " " << x[2] << "\n";
    os << "\n";

    os << "CELLS " << indices.size() << " " << 5 * indices.size() << "\n";
    for (const auto &m : indices)
      os << 4 << " " << m[0] << " " << m[1] << " " << m[2] << " " << m[3] << "\n";
    os << "\n";

    os << "CELL_TYPES " << indices.size() << "\n";
    for (int i = 0; i < indices.size(); i++) os << 10 << "\n";

    os.close();
    printf("done writing to %s\n", filename.c_str());
    return true;
  }

}  // namespace zs
