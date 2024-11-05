#include "zensim/vulkan/VkModel.hpp"

#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/vulkan/VkCommand.hpp"
#if ZS_ENABLE_OPENMP
#  include "zensim/omp/execution/ExecutionPolicy.hpp"
#endif

namespace zs {

  void VkModel::parseFromMesh(VulkanContext& ctx, const Mesh<float, 3, u32, 3>& surfs) {
    using Ti = u32;

    const auto& vs = surfs.nodes;
    const auto& is = surfs.elems;
    const auto& clrs = surfs.colors;
    const auto& uvs = surfs.uvs;
    const auto& nrms = surfs.norms;

    texturePath = surfs.texturePath;

    verts.vertexCount = vs.size();
    indexCount = is.size() * 3;

#if 1
    auto& env = ctx.env();
    auto preferredQueueType = ctx.isQueueValid(zs::vk_queue_e::dedicated_transfer)
                                  ? zs::vk_queue_e::dedicated_transfer
                                  : zs::vk_queue_e::transfer;
    auto& pool = env.pools(preferredQueueType);
    // auto copyQueue = env.pools(zs::vk_queue_e::transfer).queue;
    // auto copyQueue = pool.allQueues.back();
    auto& cmd = *pool.primaryCmd;
    auto copyQueue = ctx.getLastQueue(preferredQueueType);
    // vk::CommandBuffer cmd = pool.createCommandBuffer(vk::CommandBufferLevel::ePrimary, false,
    //     nullptr, zs::vk_cmd_usage_e::single_use);
    cmd.begin(vk::CommandBufferBeginInfo{});
    vk::BufferCopy copyRegion{};

    /// @note pos
    auto numBytes = sizeof(float) * 3 * vs.size();
    if (!stagingBuffer || numBytes > stagingBuffer.get().getSize())
      stagingBuffer = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
    stagingBuffer.get().map();
    memcpy(stagingBuffer.get().mappedAddress(), vs.data(), numBytes);
    stagingBuffer.get().unmap();
    //
    if (!verts.pos || numBytes > verts.pos.get().getSize())
      verts.pos = ctx.createBuffer(
          numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
          vk::MemoryPropertyFlagBits::eDeviceLocal);
    copyRegion.size = numBytes;
    (*cmd).copyBuffer(stagingBuffer.get(), verts.pos.get(), {copyRegion});

    /// @note colors
    std::vector<std::array<float, 3>> vals(
        vs.size(), std::array<float, 3>{0.7f, 0.7f, 0.7f});  // use 0.7 as default color
    if (!stagingColorBuffer || numBytes > stagingColorBuffer.get().getSize())
      stagingColorBuffer = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
    stagingColorBuffer.get().map();
    if (clrs.size() == vs.size())
      memcpy(stagingColorBuffer.get().mappedAddress(), clrs.data(), numBytes);
    else
      memcpy(stagingColorBuffer.get().mappedAddress(), vals.data(), numBytes);
    stagingColorBuffer.get().unmap();
    if (!verts.clr || numBytes > verts.clr.get().getSize())
      verts.clr = ctx.createBuffer(
          numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
          vk::MemoryPropertyFlagBits::eDeviceLocal);
    (*cmd).copyBuffer(stagingColorBuffer.get(), verts.clr.get(), {copyRegion});

    /// @note normals
    if (!stagingNrmBuffer || numBytes > stagingNrmBuffer.get().getSize())
      stagingNrmBuffer = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
    stagingNrmBuffer.get().map();
    if (nrms.size() == vs.size())
      memcpy(stagingNrmBuffer.get().mappedAddress(), nrms.data(), numBytes);
    else {
      compute_mesh_normal(surfs, 1.f, vals);
      memcpy(stagingNrmBuffer.get().mappedAddress(), vals.data(), numBytes);
    }
    stagingNrmBuffer.get().unmap();
    if (!verts.nrm || numBytes > verts.nrm.get().getSize())
      verts.nrm = ctx.createBuffer(
          numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
          vk::MemoryPropertyFlagBits::eDeviceLocal);
    (*cmd).copyBuffer(stagingNrmBuffer.get(), verts.nrm.get(), {copyRegion});

    /// @note uvs
    numBytes = 2 * sizeof(float) * vs.size();
    if (!stagingUVBuffer || numBytes > stagingUVBuffer.get().getSize())
      stagingUVBuffer = ctx.createStagingBuffer(
          numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    stagingUVBuffer.get().map();
    if (uvs.size() == vs.size()) {
      memcpy(stagingUVBuffer.get().mappedAddress(), uvs.data(), numBytes);
    } else {
      std::vector<std::array<float, 2>> defaultUVs{vs.size(), {0.0f, 0.0f}};
      memcpy(stagingUVBuffer.get().mappedAddress(), defaultUVs.data(), numBytes);
    }
    stagingUVBuffer.get().unmap();
    if (!verts.uv || numBytes > verts.uv.get().getSize())
      verts.uv = ctx.createBuffer(
          numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
          vk::MemoryPropertyFlagBits::eDeviceLocal);
    copyRegion.size = numBytes;
    (*cmd).copyBuffer(stagingUVBuffer.get(), verts.uv.get(), {copyRegion});

    auto numIndexBytes = sizeof(u32) * vs.size();
    if (!stagingVidBuffer || numIndexBytes > stagingVidBuffer.get().getSize())
      stagingVidBuffer
          = ctx.createStagingBuffer(numIndexBytes, vk::BufferUsageFlagBits::eTransferSrc);
    stagingVidBuffer.get().map();
    std::vector<u32> hVids(vs.size());
#  if ZS_ENABLE_OPENMP
    auto pol = omp_exec();
#  else
    auto pol = seq_exec();
#  endif
    pol(enumerate(hVids), [](u32 i, u32& dst) { dst = i; });
    memcpy(stagingVidBuffer.get().mappedAddress(), hVids.data(), numIndexBytes);
    stagingVidBuffer.get().unmap();
    if (!verts.vids || numIndexBytes > verts.vids.get().getSize())
      verts.vids = ctx.createBuffer(
          numIndexBytes,
          vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
          vk::MemoryPropertyFlagBits::eDeviceLocal);
    copyRegion.size = numIndexBytes;
    (*cmd).copyBuffer(stagingVidBuffer.get(), verts.vids.get(), {copyRegion});

    /// @note tris
    numBytes = sizeof(Ti) * 3 * is.size();
    if (!stagingIndexBuffer || numBytes > stagingIndexBuffer.get().getSize())
      stagingIndexBuffer = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
    stagingIndexBuffer.get().map();
    memcpy(stagingIndexBuffer.get().mappedAddress(), is.data(), numBytes);
    stagingIndexBuffer.get().unmap();

    if (!indices || numBytes > indices.get().getSize())
      indices = ctx.createBuffer(
          numBytes, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
          vk::MemoryPropertyFlagBits::eDeviceLocal);
    copyRegion.size = numBytes;
    (*cmd).copyBuffer(stagingIndexBuffer.get(), indices.get(), {copyRegion});

    cmd.end();
    // auto submitInfo = vk::SubmitInfo().setCommandBufferCount(1).setPCommandBuffers(&cmd);
    // vk::Fence fence = ctx.device.createFence(vk::FenceCreateInfo{}, nullptr, ctx.dispatcher);
    auto& fence = *pool.fence;
    // ctx.device.resetFences(1, &fence);
    // auto res = copyQueue.submit(1, &submitInfo, fence, ctx.dispatcher);
    cmd.submit(fence, true, true);
    fence.wait();
    // if (ctx.device.waitForFences(1, &fence, VK_TRUE, std::numeric_limits<u64>::max(),
    //                              ctx.dispatcher)
    //     != vk::Result::eSuccess)
    //   throw std::runtime_error("error waiting for fences");
    // ctx.device.destroyFence(fence, nullptr, ctx.dispatcher);
    // ctx.device.freeCommandBuffers(pool.cmdpool(zs::vk_cmd_usage_e::single_use), cmd,
    //                               ctx.dispatcher);
#else
    /// @note pos
    auto numBytes = sizeof(float) * 3 * vs.size();
    verts.pos = ctx.createBuffer(
        numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible);
    verts.pos.get().map();
    memcpy(verts.pos.get().mappedAddress(), vs.data(), numBytes);
    verts.pos.get().unmap();
    verts.pos.get().flush();

    /// @note colors
    verts.clr = ctx.createBuffer(
        numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible);
    verts.clr.get().map();
    if (clrs.size() == vs.size())
      memcpy(verts.clr.get().mappedAddress(), clrs.data(), numBytes);
    else {
      std::vector<std::array<float, 3>> vals(
          vs.size(), std::array<float, 3>{0.7f, 0.7f, 0.7f});  // use 0.7 as default color
      memcpy(verts.clr.get().mappedAddress(), vals.data(), numBytes);
    }
    verts.clr.get().unmap();
    verts.clr.get().flush();

    /// @note normals
    verts.nrm = ctx.createBuffer(
        numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible);
    verts.nrm.get().map();
    if (nrms.size() == vs.size())
      memcpy(verts.nrm.get().mappedAddress(), nrms.data(), numBytes);
    else {
      std::vector<std::array<float, 3>> vals(vs.size());
      compute_mesh_normal(surfs, 1.f, vals);
      memcpy(verts.nrm.get().mappedAddress(), vals.data(), numBytes);
    }
    verts.nrm.get().unmap();
    verts.nrm.get().flush();

    /// @note uvs
    numBytes = 2 * sizeof(float) * vs.size();
    verts.uv = ctx.createBuffer(
        numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible);
    verts.uv.get().map();
    if (uvs.size() == vs.size()) {
      memcpy(verts.uv.get().mappedAddress(), uvs.data(), numBytes);
    } else {
      std::vector<std::array<float, 2>> defaultUVs{vs.size(), {0.0f, 0.0f}};
      memcpy(verts.uv.get().mappedAddress(), defaultUVs.data(), numBytes);
    }
    verts.uv.get().unmap();
    verts.uv.get().flush();

    /// @note verts
    auto numIndexBytes = sizeof(u32) * vs.size();
    verts.vids = ctx.createBuffer(
        numIndexBytes,
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible);
    verts.vids.get().map();
    std::vector<u32> hVids(vs.size());
#  if ZS_ENABLE_OPENMP
    auto pol = omp_exec();
#  else
    auto pol = seq_exec();
#  endif
    pol(enumerate(hVids), [](u32 i, u32& dst) { dst = i; });
    memcpy(verts.vids.get().mappedAddress(), hVids.data(), numIndexBytes);
    verts.vids.get().unmap();
    verts.vids.get().flush();

    /// @note tris
    numBytes = sizeof(Ti) * 3 * is.size();
    indices = ctx.createBuffer(
        numBytes, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible);
    indices.get().map();
    memcpy(indices.get().mappedAddress(), is.data(), numBytes);
    indices.get().unmap();
    indices.get().flush();
#endif
  }

  VkModel::VkModel(VulkanContext& ctx, const Mesh<float, /*dim*/ 3, u32, /*codim*/ 3>& surfs,
                   const vec3_t& trans, const vec3_t& rotation, const vec3_t& scale)
      : translate{trans}, rotate{rotation}, scale{scale}, transform(0.0f), useTransform(false) {
    parseFromMesh(ctx, surfs);
  }

  VkModel::VkModel(VulkanContext& ctx, const Mesh<float, 3, u32, 3>& surfs,
                   const transform_t& transform)
      : transform(transform), useTransform(true) {
    parseFromMesh(ctx, surfs);
  }
}  // namespace zs