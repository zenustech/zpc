#pragma once
#include "zensim/geometry/Mesh.hpp"
#include "zensim/math/Vec.h"
#include "zensim/vulkan/VkBuffer.hpp"
#include "zensim/vulkan/VkTexture.hpp"

namespace zs {

  struct ZPC_API VkModel {
    enum draw_category_e { tri = 0, point, pick };
    using vec3_t = vec<float, 3>;
    using transform_t = vec<float, 4, 4>;
    struct Vertices {
      // vec3: pos, color, normal
      // vec2: uv
      vk::DeviceSize vertexCount;
      Owner<Buffer> pos, nrm, clr;
      Owner<Buffer> vids;  // optional
    };

    VkModel() = default;
    VkModel(VulkanContext &ctx, const Mesh<float, /*dim*/ 3, u32, /*codim*/ 3> &surfs,
            const vec3_t &translation = vec3_t::constant(0.f),
            const vec3_t &eulerXYZradians = vec3_t::constant(0.f),
            const vec3_t &scale = vec3_t::constant(1.f));
    VkModel(VkModel &&o) = default;
    VkModel &operator=(VkModel &&o) = default;
    void reset() {
      verts.pos.reset();
      verts.nrm.reset();
      verts.clr.reset();
      indices.reset();
    }
    ~VkModel() { reset(); }

    static std::vector<vk::VertexInputBindingDescription> get_binding_descriptions(
        draw_category_e e = draw_category_e::tri) noexcept {
      switch (e) {
        case draw_category_e::tri:
          return std::vector<vk::VertexInputBindingDescription>{
              {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
              {1, /*normal*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
              {2, /*color*/ sizeof(float) * 3, vk::VertexInputRate::eVertex}};
        case draw_category_e::point:
          // radius is specified through push constant
          return std::vector<vk::VertexInputBindingDescription>{
              {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
              {1, /*color*/ sizeof(float) * 3, vk::VertexInputRate::eVertex}};
        case draw_category_e::pick:
          return std::vector<vk::VertexInputBindingDescription>{
              {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
              {1, /*vid*/ sizeof(u32), vk::VertexInputRate::eVertex}};
        default:;
      }
      return {};
    }
    static std::vector<vk::VertexInputAttributeDescription> get_attribute_descriptions(
        draw_category_e e = draw_category_e::tri) noexcept {
      switch (e) {
        case draw_category_e::tri:
          return std::vector<vk::VertexInputAttributeDescription>{
              {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
              {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
              {/*location*/ 2, /*binding*/ 2, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0}};
        case draw_category_e::point:
          return std::vector<vk::VertexInputAttributeDescription>{
              {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
              {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0}};
        case draw_category_e::pick:
          return std::vector<vk::VertexInputAttributeDescription>{
              {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
              {/*location*/ 1, /*binding*/ 1, vk::Format::eR32Uint, /*offset*/ (u32)0}};
        default:;
      }
      return {};
    }

    void draw(const vk::CommandBuffer &cmd, draw_category_e e = draw_category_e::tri) const {
      switch (e) {
        case draw_category_e::tri:
          cmd.drawIndexed(/*index count*/ indexCount, /*instance count*/ 1,
                          /*first index*/ 0, /*vertex offset*/ 0,
                          /*first instance*/ 0, indices.get().ctx.dispatcher);
          break;
        case draw_category_e::point:
        case draw_category_e::pick:
          cmd.draw(/*vertex count*/ verts.vertexCount, /*instance count*/ 1,
                   /*first vertex*/ 0, /*first instance*/ 0, indices.get().ctx.dispatcher);
          break;
        default:;
      }
    }

    void bind(const vk::CommandBuffer &cmd, draw_category_e e = draw_category_e::tri) const {
      switch (e) {
        case draw_category_e::tri: {
          std::array<vk::Buffer, 3> buffers{verts.pos.get(), verts.nrm.get(), verts.clr.get()};
          std::array<vk::DeviceSize, 3> offsets{0, 0, 0};
          cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets,
                                verts.pos.get().ctx.dispatcher);
          cmd.bindIndexBuffer({(vk::Buffer)indices.get()}, /*offset*/ (u32)0,
                              vk::IndexType::eUint32, indices.get().ctx.dispatcher);
        } break;
        case draw_category_e::point: {
          std::array<vk::Buffer, 2> buffers{verts.pos.get(), verts.clr.get()};
          std::array<vk::DeviceSize, 2> offsets{0, 0};
          cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets,
                                verts.pos.get().ctx.dispatcher);
        } break;
        case draw_category_e::pick: {
          std::array<vk::Buffer, 2> buffers{verts.pos.get(), verts.vids.get()};
          std::array<vk::DeviceSize, 2> offsets{0, 0};
          cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets,
                                verts.pos.get().ctx.dispatcher);
        } break;
        default:;
      }
    }

    bool isParticle() const noexcept { return indexCount == 0; }

    Vertices verts;
    vk::DeviceSize indexCount;
    Owner<Buffer> indices;
    vec3_t scale, rotate, translate;
  };

}  // namespace zs