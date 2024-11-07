#pragma once
#include "zensim/geometry/Mesh.hpp"
#include "zensim/math/Vec.h"
#include "zensim/vulkan/VkBuffer.hpp"
#include "zensim/vulkan/VkTexture.hpp"

namespace zs {

  struct ZPC_API VkModel {
    enum draw_category_e { tri = 0, line, point, pick, normal };
    using vec3_t = vec<float, 3>;
    using transform_t = vec<float, 4, 4>;
    struct Vertices {
      // vec3: pos, color, normal
      // vec2: uv
      vk::DeviceSize vertexCount;
      Owner<Buffer> pos, nrm, clr, uv;
      Owner<Buffer> vids;  // optional
    };
    // staging buffers
    Owner<Buffer> stagingBuffer, stagingNrmBuffer, stagingColorBuffer, stagingUVBuffer,
        stagingVidBuffer, stagingIndexBuffer;

    Buffer &getColorBuffer() { return verts.clr.get(); }

    VkModel() = default;
    VkModel(VulkanContext &ctx, const Mesh<float, /*dim*/ 3, u32, /*codim*/ 3> &surfs,
            const vec3_t &translation = vec3_t::constant(0.f),
            const vec3_t &eulerXYZradians = vec3_t::constant(0.f),
            const vec3_t &scale = vec3_t::constant(1.f));
    VkModel(VulkanContext &ctx, const Mesh<float, 3, u32, 3> &surfs, const transform_t &transform);
    VkModel(VkModel &&o) = default;
    VkModel &operator=(VkModel &&o) = default;
    void parseFromMesh(VulkanContext &, const Mesh<float, 3, u32, 3> &surfs);
    void reset() {
      verts.pos.reset();
      verts.nrm.reset();
      verts.clr.reset();
      verts.uv.reset();
      verts.vids.reset();
      indices.reset();
      texturePath.clear();
    }
    ~VkModel() { reset(); }

    static std::vector<vk::VertexInputBindingDescription> get_binding_descriptions(
        draw_category_e e = draw_category_e::tri) noexcept;
    static std::vector<vk::VertexInputAttributeDescription> get_attribute_descriptions(
        draw_category_e e = draw_category_e::tri) noexcept;
    void draw(const vk::CommandBuffer &cmd, draw_category_e e = draw_category_e::tri) const;
    void bind(const vk::CommandBuffer &cmd, draw_category_e e = draw_category_e::tri) const;

    bool isParticle() const noexcept { return indexCount == 0; }
    bool isValid() const noexcept { return verts.vertexCount > 0; }

    VulkanContext *pCtx() noexcept {
      if (verts.pos) return verts.pos.get().pCtx();
      return nullptr;
    }
    const VulkanContext *pCtx() const noexcept {
      if (verts.pos) return verts.pos.get().pCtx();
      return nullptr;
    }

    Vertices verts;
    vk::DeviceSize indexCount;
    Owner<Buffer> indices;
    vec3_t scale, rotate, translate;
    transform_t transform;
    bool useTransform = false;
    std::string texturePath;
  };

}  // namespace zs