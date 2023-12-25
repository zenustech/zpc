#pragma once
#include "zensim/geometry/Mesh.hpp"
#include "zensim/math/Vec.h"
#include "zensim/vulkan/VkBuffer.hpp"
#include "zensim/vulkan/VkTexture.hpp"

namespace zs {

  struct VkModel {
    using transform_t = vec<float, 4, 4>;
    struct Vertices {
      // vec3: pos, color, normal
      // vec2: uv
      vk::DeviceSize vertexCount;
      Owner<Buffer> pos, nrm, clr;
      Owner<Buffer> vids;  // optional
    };

    VkModel() = default;
    template <typename Ti> VkModel(VulkanContext &ctx,
                                   const Mesh<float, /*dim*/ 3, Ti, /*codim*/ 3> &surfs,
                                   const transform_t &trans = transform_t::identity());
    VkModel(VkModel &&o) = default;
    VkModel &operator=(VkModel &&o) = default;
    void reset() {
      verts.pos.reset();
      verts.nrm.reset();
      verts.clr.reset();
      indices.reset();
    }
    ~VkModel() { reset(); }

    static std::vector<vk::VertexInputBindingDescription> get_binding_descriptions() noexcept {
      return std::vector<vk::VertexInputBindingDescription>{
          {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
          {1, /*normal*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
          {2, /*color*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
          {3, /*vid*/ sizeof(u32), vk::VertexInputRate::eVertex}};
    }
    static std::vector<vk::VertexInputAttributeDescription> get_attribute_descriptions() noexcept {
      return std::vector<vk::VertexInputAttributeDescription>{
          {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
          {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
          {/*location*/ 2, /*binding*/ 2, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
          {/*location*/ 3, /*binding*/ 3, vk::Format::eR32Uint, /*offset*/ (u32)0},
      };
    }

    void draw(const vk::CommandBuffer &cmd) const {
      cmd.drawIndexed(/*index count*/ indexCount, /*instance count*/ 1,
                      /*first index*/ 0, /*vertex offset*/ 0,
                      /*first instance*/ 0, indices.get().ctx.dispatcher);
    }

    void bind(const vk::CommandBuffer &cmd) const {
      vk::Buffer buffers[] = {verts.pos.get(), verts.nrm.get(), verts.clr.get(), verts.vids.get()};
      vk::DeviceSize offsets[] = {0, 0, 0, 0};
      cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets, verts.pos.get().ctx.dispatcher);

      cmd.bindIndexBuffer({(vk::Buffer)indices.get()}, /*offset*/ (u32)0, vk::IndexType::eUint32,
                          indices.get().ctx.dispatcher);
    }

    Vertices verts;
    vk::DeviceSize indexCount;
    Owner<Buffer> indices;
    transform_t transform;
  };

  ZPC_FWD_DECL_FUNC VkModel::VkModel(VulkanContext &ctx, const Mesh<float, 3, u32, 3> &surfs,
                                     const transform_t &trans);
  ZPC_FWD_DECL_FUNC VkModel::VkModel(VulkanContext &ctx, const Mesh<float, 3, i32, 3> &surfs,
                                     const transform_t &trans);

}  // namespace zs