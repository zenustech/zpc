#pragma once
#include "zensim/vulkan/VkBuffer.hpp"

namespace zs {

  struct VkModel {
    struct Vertices {
      // vec3: pos, color, normal
      // vec2: uv
      std::unique_ptr<Buffer> pos, nrm, clr;
    };

    template <typename Ti>
    VkModel(VulkanContext &ctx, const Mesh<float, /*dim*/ 3, Ti, /*codim*/ 3> &surfs) {
      const auto &vs = surfs.nodes;
      const auto &is = surfs.elems;

      auto &env = ctx.env();
      auto &pool = env.pools(zs::vk_queue_e::graphics);
      // auto copyQueue = env.pools(zs::vk_queue_e::transfer).queue;
      auto copyQueue = pool.queue;
      vk::CommandBuffer cmd = pool.createCommandBuffer(vk::CommandBufferLevel::ePrimary, false,
                                                       nullptr, zs::vk_cmd_usage_e::single_use);
      cmd.begin(vk::CommandBufferBeginInfo{});
      vk::BufferCopy copyRegion{};

      /// @note pos
      auto numBytes = sizeof(float) * 3 * vs.size();
      auto stagingBuffer = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
      stagingBuffer.map();
      memcpy(stagingBuffer.mappedAddress(), vs.data(), numBytes);
      //
      verts.pos = std::make_unique<Buffer>(ctx.createBuffer(
          numBytes,
          vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst));
      copyRegion.size = numBytes;
      cmd.copyBuffer(stagingBuffer, *verts.pos, {copyRegion});

      /// @note colors
      std::vector<std::array<float, 3>> vals(vs.size(), std::array<float, 3>{1.f, 1.f, 1.f});
      memcpy(stagingBuffer.mappedAddress(), vals.data(), numBytes);
      verts.clr = std::make_unique<Buffer>(ctx.createBuffer(
          numBytes,
          vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst));
      cmd.copyBuffer(stagingBuffer, *verts.clr, {copyRegion});

      /// @note normals
      compute_mesh_normal(surfs, 1.f, vals);
      memcpy(stagingBuffer.mappedAddress(), vals.data(), numBytes);
      verts.nrm = std::make_unique<Buffer>(ctx.createBuffer(
          numBytes,
          vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst));
      cmd.copyBuffer(stagingBuffer, *verts.nrm, {copyRegion});

      /// @note tris
      numBytes = sizeof(Ti) * 3 * is.size();
      if ((sizeof(Ti) != sizeof(float) || alignof(Ti) != alignof(float))
          || vs.size() != is.size()) {
        stagingBuffer.unmap();
        stagingBuffer.moveAssign(
            ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc));
        stagingBuffer.map();
      }
      memcpy(stagingBuffer.mappedAddress(), is.data(), numBytes);
      indices = std::make_unique<Buffer>(ctx.createBuffer(
          numBytes, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst));
      copyRegion.size = numBytes;
      cmd.copyBuffer(stagingBuffer, *indices, {copyRegion});

      cmd.end();
      auto submitInfo = vk::SubmitInfo().setCommandBufferCount(1).setPCommandBuffers(&cmd);
      vk::Fence fence = ctx.device.createFence(vk::FenceCreateInfo{}, nullptr, ctx.dispatcher);
      // ctx.device.resetFences(1, &fence);
      auto res = copyQueue.submit(1, &submitInfo, fence, ctx.dispatcher);
      if (ctx.device.waitForFences(1, &fence, VK_TRUE, std::numeric_limits<u64>::max(),
                                   ctx.dispatcher)
          != vk::Result::eSuccess)
        throw std::runtime_error("error waiting for fences");
      ctx.device.destroyFence(fence, nullptr, ctx.dispatcher);
      ctx.device.freeCommandBuffers(pool.cmdpool(zs::vk_cmd_usage_e::single_use), cmd,
                                    ctx.dispatcher);

      stagingBuffer.unmap();
    }

    std::vector<vk::VertexInputBindingDescription> getBindingDescriptions() const noexcept {
      return std::vector<vk::VertexInputBindingDescription>{
          {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
          {1, /*normal*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
          {2, /*color*/ sizeof(float) * 3, vk::VertexInputRate::eVertex}};
    }
    std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions() const noexcept {
      return std::vector<vk::VertexInputAttributeDescription>{
          {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
          {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
          {/*location*/ 2, /*binding*/ 2, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0}};
    }

    Vertices verts;
    std::unique_ptr<Buffer> indices;
  };

}  // namespace zs