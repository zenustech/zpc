#include "MeshIO.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "zensim/zpc_tpls/tinyobj/tiny_obj_loader.h"

namespace zs {

  static_assert(is_same_v<tinyobj::real_t, float>,
                "'real_t' (from tinyobj) is assumed to be float.");

  bool load_obj(std::string_view file, std::vector<std::array<float, 3>> *pos,
                std::vector<std::array<float, 3>> *nrm, std::vector<std::array<float, 2>> *uv,
                std::vector<std::array<u32, 3>> *tris) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, file.data(), nullptr,
                                /*triangulate*/ true);

    if (!warn.empty()) {
      printf("WARN: %s\n", warn.data());
    }
    if (!err.empty()) {
      fprintf(stderr, "ERR: %s\n", err.data());
    }

    if (!ret) {
      return false;
    }

    auto vertOffset = pos->size();
    auto nrmOffset = nrm->size();
    auto uvOffset = uv->size();
    auto triOffset = tris->size();
    // pos
    auto nVerts = attrib.vertices.size() / 3;
    if (pos) {
      pos->resize(vertOffset + nVerts);
      std::memcpy(pos->data() + vertOffset, attrib.vertices.data(),
                  attrib.vertices.size() * sizeof(float));
    }
    // nrm
    auto nNrms = attrib.normals.size() / 3;
    if (nrm) {
      nrm->resize(nrmOffset + nVerts, std::array<float, 3>{0.f, 0.f, 0.f});
    }
    // uv
    auto nUvs = attrib.texcoords.size() / 2;
    if (uv) {
      uv->resize(uvOffset + nVerts, std::array<float, 2>{0.f, 0.f});
    }
    ///
    auto nTris = 0;
    for (size_t i = 0; i < shapes.size(); ++i) {
      nTris += shapes[i].mesh.num_face_vertices.size();
    }
    // printf("num tris: %d, num pos: %d, num nrms: %d, num uvs: %d\n", (int)nTris, (int)nVerts,
    //       (int)nNrms, (int)nUvs);
    if (tris) {
      tris->resize(triOffset + nTris);
      size_t globalTriOffset = 0;
      for (size_t i = 0; i < shapes.size(); ++i) {
        const auto &shape = shapes[i];

        size_t indexOffset = 0;

        // For each face
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
          size_t fnum = shape.mesh.num_face_vertices[f];
          assert(fnum == 3);

          // For each vertex in the face
          for (size_t v = 0; v < fnum; v++) {
            tinyobj::index_t idx = shape.mesh.indices[indexOffset + v];
            u32 vId = idx.vertex_index;
            u32 nId = idx.normal_index;
            u32 tId = idx.texcoord_index;

            (*tris)[triOffset + globalTriOffset + f][v] = vId;  // normal_index, texcoord_index

            if (nrm && idx.normal_index != -1)
              (*nrm)[nrmOffset + vId]
                  = std::array<float, 3>{attrib.normals[nId * 3], attrib.normals[nId * 3 + 1],
                                         attrib.normals[nId * 3 + 2]};

            if (uv && idx.texcoord_index != -1)
              (*uv)[uvOffset + vId]
                  = std::array<float, 2>{attrib.texcoords[tId * 2], attrib.texcoords[tId * 2 + 1]};
          }

          indexOffset += fnum;
        }

        globalTriOffset += shape.mesh.num_face_vertices.size();
      }
    }
    if (nNrms == 0) {
      compute_mesh_normal(*pos, vertOffset, nVerts, *tris, triOffset, nTris,
                          nrm->data() + nrmOffset);
    }

    return true;
  }

}  // namespace zs