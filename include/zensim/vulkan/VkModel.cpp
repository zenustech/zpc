#include "zensim/vulkan/VkModel.hpp"

#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

  template <typename Ti>
  VkModel::VkModel(VulkanContext &ctx, const Mesh<float, /*dim*/ 3, Ti, /*codim*/ 3> &surfs,
                   const transform_t &trans)
      : transform{trans} {
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
    verts.pos = ctx.createBuffer(
        numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst);
    copyRegion.size = numBytes;
    cmd.copyBuffer(stagingBuffer, verts.pos.get(), {copyRegion});

    /// @note colors
    std::vector<std::array<float, 3>> vals(vs.size(), std::array<float, 3>{0.7f, 0.7f, 0.7f});
    auto stagingColorBuffer
        = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
    stagingColorBuffer.map();
    memcpy(stagingColorBuffer.mappedAddress(), vals.data(), numBytes);
    stagingColorBuffer.unmap();
    verts.clr = ctx.createBuffer(
        numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst);
    cmd.copyBuffer(stagingColorBuffer, verts.clr.get(), {copyRegion});

    /// @note normals
    compute_mesh_normal(surfs, 1.f, vals);
    auto stagingNrmBuffer
        = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
    stagingNrmBuffer.map();
    memcpy(stagingNrmBuffer.mappedAddress(), vals.data(), numBytes);
    stagingNrmBuffer.unmap();
    verts.nrm = ctx.createBuffer(
        numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst);
    cmd.copyBuffer(stagingNrmBuffer, verts.nrm.get(), {copyRegion});

    auto numIndexBytes = sizeof(u32) * vs.size();
    auto stagingVidBuffer
        = ctx.createStagingBuffer(numIndexBytes, vk::BufferUsageFlagBits::eTransferSrc);
    stagingVidBuffer.map();
    std::vector<u32> hVids(vs.size());
    auto pol = seq_exec();
    pol(enumerate(hVids), [](u32 i, u32 &dst) { dst = i; });
    memcpy(stagingVidBuffer.mappedAddress(), hVids.data(), numIndexBytes);
    stagingVidBuffer.unmap();
    verts.vids = ctx.createBuffer(numIndexBytes, vk::BufferUsageFlagBits::eVertexBuffer
                                                     | vk::BufferUsageFlagBits::eTransferDst);
    copyRegion.size = numIndexBytes;
    cmd.copyBuffer(stagingVidBuffer, verts.vids.get(), {copyRegion});

    /// @note tris
    numBytes = sizeof(Ti) * 3 * is.size();
    auto stagingIndexBuffer
        = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
    stagingIndexBuffer.map();
    memcpy(stagingIndexBuffer.mappedAddress(), is.data(), numBytes);
    stagingIndexBuffer.unmap();

    indices = ctx.createBuffer(
        numBytes, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst);
    copyRegion.size = numBytes;
    cmd.copyBuffer(stagingIndexBuffer, indices.get(), {copyRegion});

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

  ZPC_INSTANTIATE VkModel::VkModel(VulkanContext &ctx,
                                   const Mesh<float, /*dim*/ 3, u32, /*codim*/ 3> &surfs,
                                   const transform_t &trans);
  ZPC_INSTANTIATE VkModel::VkModel(VulkanContext &ctx,
                                   const Mesh<float, /*dim*/ 3, i32, /*codim*/ 3> &surfs,
                                   const transform_t &trans);

}  // namespace zs