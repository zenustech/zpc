#include "zensim/vulkan/VkModel.hpp"

#include "zensim/execution/ExecutionPolicy.hpp"
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

      verts.vertexCount = vs.size();
      indexCount = is.size() * 3;

      auto& env = ctx.env();
      auto& pool = env.pools(zs::vk_queue_e::transfer);
      // auto copyQueue = env.pools(zs::vk_queue_e::transfer).queue;
      auto copyQueue = pool.allQueues.back();
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
          numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible);
      copyRegion.size = numBytes;
      cmd.copyBuffer(stagingBuffer, verts.pos.get(), { copyRegion });

      /// @note colors
      std::vector<std::array<float, 3>> vals(vs.size(), std::array<float, 3>{0.7f, 0.7f, 0.7f}); // use 0.7 as default color
      auto stagingColorBuffer = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
      stagingColorBuffer.map();
      memcpy(stagingColorBuffer.mappedAddress(), clrs.size() > 0 ? clrs.data() : vals.data(), numBytes);
      stagingColorBuffer.unmap();
      verts.clr = ctx.createBuffer(
          numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible);
      cmd.copyBuffer(stagingColorBuffer, verts.clr.get(), { copyRegion });

      /// @note normals
      compute_mesh_normal(surfs, 1.f, vals);
      auto stagingNrmBuffer
          = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
      stagingNrmBuffer.map();
      memcpy(stagingNrmBuffer.mappedAddress(), vals.data(), numBytes);
      stagingNrmBuffer.unmap();
      verts.nrm = ctx.createBuffer(
          numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst);
      cmd.copyBuffer(stagingNrmBuffer, verts.nrm.get(), { copyRegion });

      /// @note uvs
      numBytes = 2 * sizeof(float) * vs.size();
      auto stagingUVBuffer = ctx.createStagingBuffer(
          numBytes,
          vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferSrc
      );
      stagingUVBuffer.map();
      if (uvs.size() == vs.size()) {
          memcpy(stagingUVBuffer.mappedAddress(), uvs.data(), numBytes);
      } else {
          std::vector<std::array<float, 2>> defaultUVs{ vs.size(), {0.0f, 0.0f} };
          memcpy(stagingUVBuffer.mappedAddress(), defaultUVs.data(), numBytes);
      }
      stagingUVBuffer.unmap();
      verts.uv = ctx.createBuffer(
          numBytes,
          vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst
      );
      copyRegion.size = numBytes;
      cmd.copyBuffer(stagingUVBuffer, verts.uv.get(), { copyRegion });

      auto numIndexBytes = sizeof(u32) * vs.size();
      auto stagingVidBuffer
          = ctx.createStagingBuffer(numIndexBytes, vk::BufferUsageFlagBits::eTransferSrc);
      stagingVidBuffer.map();
      std::vector<u32> hVids(vs.size());
#if ZS_ENABLE_OPENMP
      auto pol = omp_exec();
#else
      auto pol = seq_exec();
#endif
      pol(enumerate(hVids), [](u32 i, u32& dst) { dst = i; });
      memcpy(stagingVidBuffer.mappedAddress(), hVids.data(), numIndexBytes);
      stagingVidBuffer.unmap();
      verts.vids = ctx.createBuffer(numIndexBytes, vk::BufferUsageFlagBits::eVertexBuffer
          | vk::BufferUsageFlagBits::eTransferDst);
      copyRegion.size = numIndexBytes;
      cmd.copyBuffer(stagingVidBuffer, verts.vids.get(), { copyRegion });

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
      cmd.copyBuffer(stagingIndexBuffer, indices.get(), { copyRegion });

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

  VkModel::VkModel(VulkanContext &ctx, const Mesh<float, /*dim*/ 3, u32, /*codim*/ 3> &surfs,
                   const vec3_t &trans, const vec3_t &rotation, const vec3_t &scale)
      : translate{trans}, rotate{rotation}, scale{scale}, transform(0.0f), useTransform(false) {
      parseFromMesh(ctx, surfs);
  }

  VkModel::VkModel(VulkanContext& ctx, const Mesh<float, 3, u32, 3>& surfs, const transform_t& transform)
    :transform(transform), useTransform(true) {
      parseFromMesh(ctx, surfs);
  }
}  // namespace zs