#pragma once
#include "zensim/vulkan/VkBuffer.hpp"

namespace zs {

  struct VkModel {
    struct Vertices {
      // vec3: pos, color, normal
      // vec2: uv
      vk::DeviceSize vertexCount;
      std::unique_ptr<Buffer> pos, nrm, clr;
    };

    template <typename Ti>
    VkModel(VulkanContext &ctx, const Mesh<float, /*dim*/ 3, Ti, /*codim*/ 3> &surfs) {
      static_assert(sizeof(Ti) == sizeof(u32) && alignof(Ti) == alignof(u32),
                    "index type should be u32-alike");

      const auto &vs = surfs.nodes;
      const auto &is = surfs.elems;

      verts.vertexCount = vs.size();
      indexCount = is.size() * 3;

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
      stagingBuffer.unmap();
      //
      verts.pos = std::make_unique<Buffer>(ctx.createBuffer(
          numBytes,
          vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst));
      copyRegion.size = numBytes;
      cmd.copyBuffer(stagingBuffer, *verts.pos, {copyRegion});

      /// @note colors
      std::vector<std::array<float, 3>> vals(vs.size(), std::array<float, 3>{1.f, 1.f, 1.f});
      auto stagingColorBuffer
          = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
      stagingColorBuffer.map();
      memcpy(stagingColorBuffer.mappedAddress(), vals.data(), numBytes);
      stagingColorBuffer.unmap();
      verts.clr = std::make_unique<Buffer>(ctx.createBuffer(
          numBytes,
          vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst));
      cmd.copyBuffer(stagingColorBuffer, *verts.clr, {copyRegion});

      /// @note normals
      compute_mesh_normal(surfs, 1.f, vals);
      auto stagingNrmBuffer
          = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
      stagingNrmBuffer.map();
      memcpy(stagingNrmBuffer.mappedAddress(), vals.data(), numBytes);
      stagingNrmBuffer.unmap();
      verts.nrm = std::make_unique<Buffer>(ctx.createBuffer(
          numBytes,
          vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst));
      cmd.copyBuffer(stagingNrmBuffer, *verts.nrm, {copyRegion});

      /// @note tris
      numBytes = sizeof(Ti) * 3 * is.size();
      auto stagingIndexBuffer
          = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
      stagingIndexBuffer.map();
      memcpy(stagingIndexBuffer.mappedAddress(), is.data(), numBytes);
      stagingIndexBuffer.unmap();

      indices = std::make_unique<Buffer>(ctx.createBuffer(
          numBytes, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst));
      copyRegion.size = numBytes;
      cmd.copyBuffer(stagingIndexBuffer, *indices, {copyRegion});

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
    }
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
          {2, /*color*/ sizeof(float) * 3, vk::VertexInputRate::eVertex}};
    }
    static std::vector<vk::VertexInputAttributeDescription> get_attribute_descriptions() noexcept {
      return std::vector<vk::VertexInputAttributeDescription>{
          {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
          {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
          {/*location*/ 2, /*binding*/ 2, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0}};
    }

    void draw(const vk::CommandBuffer &cmd) {
      cmd.drawIndexed(/*index count*/ indexCount, /*instance count*/ 1,
                      /*first index*/ 0, /*vertex offset*/ 0,
                      /*first instance*/ 0, indices->ctx.dispatcher);
    }

    void bind(const vk::CommandBuffer &cmd) {
      vk::Buffer buffers[] = {*verts.pos, *verts.nrm, *verts.clr};
      vk::DeviceSize offsets[] = {0, 0, 0};
      cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets, verts.pos->ctx.dispatcher);

      cmd.bindIndexBuffer({*indices}, /*offset*/ 0, vk::IndexType::eUint32,
                          indices->ctx.dispatcher);
    }

    Vertices verts;
    vk::DeviceSize indexCount;
    std::unique_ptr<Buffer> indices;
  };

}  // namespace zs