#include "zensim/vulkan/VkModel.hpp"

#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/vulkan/VkCommand.hpp"
#if ZS_ENABLE_OPENMP
#  include "zensim/omp/execution/ExecutionPolicy.hpp"
#endif

namespace zs {

  void VkModel::parseFromMesh(VulkanContext& ctx, const Mesh<float, 3, u32, 3>& surfs) {
    auto& env = ctx.env();
    auto preferredQueueType = ctx.isQueueValid(zs::vk_queue_e::dedicated_transfer)
                                  ? zs::vk_queue_e::dedicated_transfer
                                  : zs::vk_queue_e::transfer;
    auto& pool = env.pools(preferredQueueType);
    // auto copyQueue = env.pools(zs::vk_queue_e::transfer).queue;
    // auto copyQueue = pool.allQueues.back();
    // auto& cmd = *pool.primaryCmd;
    auto cmd = ctx.createCommandBuffer(vk_cmd_usage_e::single_use, preferredQueueType, false);
    auto copyQueue = ctx.getLastQueue(preferredQueueType);
    // vk::CommandBuffer cmd = pool.createCommandBuffer(vk::CommandBufferLevel::ePrimary, false,
    //     nullptr, zs::vk_cmd_usage_e::single_use);
    cmd.begin(vk::CommandBufferBeginInfo{});

    /// actual vk command recording
    parseFromMesh(ctx, *cmd, surfs);

    cmd.end();
    // vk::Fence fence = ctx.device.createFence(vk::FenceCreateInfo{}, nullptr, ctx.dispatcher);
    auto& fence = *pool.fence;
#if 0
    cmd.submit(fence, true, true);
    fence.wait();
#else
    ctx.device.resetFences({fence}, ctx.dispatcher);
    vk::CommandBuffer cmd_ = *cmd;
    auto submitInfo = vk::SubmitInfo().setCommandBufferCount(1).setPCommandBuffers(&cmd_);
    auto res = copyQueue.submit(1, &submitInfo, fence, ctx.dispatcher);
    fence.wait();
#endif
  }
  void VkModel::parseFromMesh(VulkanContext& ctx, vk::CommandBuffer cmd,
                              const Mesh<float, 3, u32, 3>& surfs) {
    using Ti = u32;

    const auto& vs = surfs.nodes;
    const auto& is = surfs.elems;
    const auto& clrs = surfs.colors;
    const auto& uvs = surfs.uvs;
    const auto& nrms = surfs.norms;
    const auto& tans = surfs.tans;
    const auto& texids = surfs.texids;

    texturePath = surfs.texturePath;

    verts.vertexCount = vs.size();
    indexCount = is.size() * 3;

    if (indexCount == 0) return;

#if 1
    vk::BufferCopy copyRegion{};

    /// @note pos
    auto numBytes = sizeof(float) * 3 * vs.size();
    if (!stagingBuffer || numBytes > stagingBuffer.get().getSize()) {
      if (stagingBuffer) stagingBuffer.get().unmap();
      stagingBuffer = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
      stagingBuffer.get().map();
    }
    memcpy(stagingBuffer.get().mappedAddress(), vs.data(), numBytes);
    //
    if (!verts.pos || numBytes > verts.pos.get().getSize())
      verts.pos = ctx.createBuffer(
          numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
          vk::MemoryPropertyFlagBits::eDeviceLocal);
    copyRegion.size = numBytes;
    cmd.copyBuffer(stagingBuffer.get(), verts.pos.get(), {copyRegion});

    /// @note colors
    std::vector<std::array<float, 3>> vals(
        vs.size(), std::array<float, 3>{0.7f, 0.7f, 0.7f});  // use 0.7 as default color
    if (!stagingColorBuffer || numBytes > stagingColorBuffer.get().getSize()) {
      if (stagingColorBuffer) stagingColorBuffer.get().unmap();
      stagingColorBuffer = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
      stagingColorBuffer.get().map();
    }
    if (clrs.size() == vs.size())
      memcpy(stagingColorBuffer.get().mappedAddress(), clrs.data(), numBytes);
    else
      memcpy(stagingColorBuffer.get().mappedAddress(), vals.data(), numBytes);
    if (!verts.clr || numBytes > verts.clr.get().getSize())
      verts.clr = ctx.createBuffer(
          numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
          vk::MemoryPropertyFlagBits::eDeviceLocal);
    cmd.copyBuffer(stagingColorBuffer.get(), verts.clr.get(), {copyRegion});

    /// @note normals
    if (!stagingNrmBuffer || numBytes > stagingNrmBuffer.get().getSize()) {
      if (stagingNrmBuffer) stagingNrmBuffer.get().unmap();
      stagingNrmBuffer = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
      stagingNrmBuffer.get().map();
    }
    if (nrms.size() == vs.size())
      memcpy(stagingNrmBuffer.get().mappedAddress(), nrms.data(), numBytes);
    else {
      compute_mesh_normal(surfs, 1.f, vals);
      memcpy(stagingNrmBuffer.get().mappedAddress(), vals.data(), numBytes);
    }
    if (!verts.nrm || numBytes > verts.nrm.get().getSize())
      verts.nrm = ctx.createBuffer(
          numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
          vk::MemoryPropertyFlagBits::eDeviceLocal);
    cmd.copyBuffer(stagingNrmBuffer.get(), verts.nrm.get(), {copyRegion});

    /// @note tangents
    if (!stagingTangentBuffer || numBytes > stagingTangentBuffer.get().getSize()) {
      if (stagingTangentBuffer) stagingTangentBuffer.get().unmap();
      stagingTangentBuffer = ctx.createStagingBuffer(
          numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferSrc);
      stagingTangentBuffer.get().map();
    }
    if (tans.size() == vs.size()) {
      memcpy(stagingTangentBuffer.get().mappedAddress(), tans.data(), numBytes);
    } else {
      std::vector<std::array<float, 3>> defaultTans{vs.size(), {0.0f, 0.0f, 0.f}};
      memcpy(stagingTangentBuffer.get().mappedAddress(), defaultTans.data(), numBytes);
    }
    if (!verts.tan || numBytes > verts.tan.get().getSize())
      verts.tan = ctx.createBuffer(
          numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
          vk::MemoryPropertyFlagBits::eDeviceLocal);
    copyRegion.size = numBytes;
    cmd.copyBuffer(stagingTangentBuffer.get(), verts.tan.get(), {copyRegion});

    /// @note uvs
    numBytes = 2 * sizeof(float) * vs.size();
    if (!stagingUVBuffer || numBytes > stagingUVBuffer.get().getSize()) {
      if (stagingUVBuffer) stagingUVBuffer.get().unmap();
      stagingUVBuffer = ctx.createStagingBuffer(
          numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferSrc);
      stagingUVBuffer.get().map();
    }
    if (uvs.size() == vs.size()) {
      memcpy(stagingUVBuffer.get().mappedAddress(), uvs.data(), numBytes);
    } else {
      std::vector<std::array<float, 2>> defaultUVs{vs.size(), {0.0f, 0.0f}};
      memcpy(stagingUVBuffer.get().mappedAddress(), defaultUVs.data(), numBytes);
    }
    if (!verts.uv || numBytes > verts.uv.get().getSize())
      verts.uv = ctx.createBuffer(
          numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
          vk::MemoryPropertyFlagBits::eDeviceLocal);
    copyRegion.size = numBytes;
    cmd.copyBuffer(stagingUVBuffer.get(), verts.uv.get(), {copyRegion});

    auto numIndexBytes = sizeof(u32) * vs.size();
    if (!stagingVidBuffer || numIndexBytes > stagingVidBuffer.get().getSize()) {
      if (stagingVidBuffer) stagingVidBuffer.get().unmap();
      stagingVidBuffer
          = ctx.createStagingBuffer(numIndexBytes, vk::BufferUsageFlagBits::eTransferSrc);
      stagingVidBuffer.get().map();
    }
    std::vector<u32> hVids(vs.size());
#  if ZS_ENABLE_OPENMP
    auto pol = omp_exec();
#  else
    auto pol = seq_exec();
#  endif
    pol(enumerate(hVids), [](u32 i, u32& dst) { dst = i; });
    memcpy(stagingVidBuffer.get().mappedAddress(), hVids.data(), numIndexBytes);
    if (!verts.vids || numIndexBytes > verts.vids.get().getSize())
      verts.vids = ctx.createBuffer(
          numIndexBytes,
          vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
          vk::MemoryPropertyFlagBits::eDeviceLocal);
    copyRegion.size = numIndexBytes;
    cmd.copyBuffer(stagingVidBuffer.get(), verts.vids.get(), {copyRegion});

    /// @note tris
    numBytes = sizeof(Ti) * 3 * is.size();
    if (!stagingIndexBuffer || numBytes > stagingIndexBuffer.get().getSize()) {
      if (stagingIndexBuffer) stagingIndexBuffer.get().unmap();
      stagingIndexBuffer = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
      stagingIndexBuffer.get().map();
    }
    memcpy(stagingIndexBuffer.get().mappedAddress(), is.data(), numBytes);

    if (!indices || numBytes > indices.get().getSize())
      indices = ctx.createBuffer(
          numBytes, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
          vk::MemoryPropertyFlagBits::eDeviceLocal);
    copyRegion.size = numBytes;
    cmd.copyBuffer(stagingIndexBuffer.get(), indices.get(), {copyRegion});

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

    updatePointTextureId(cmd, reinterpret_cast<const int*>(texids.data()),
                         texids.size() * sizeof(RM_REF_T(texids)::value_type));  // init texture ids
  }

  void VkModel::updateAttribsFromMesh(const Mesh<float, 3, u32, 3>& surfs, bool updatePos,
                                      bool updateColor, bool updateUv, bool updateNormal,
                                      bool updateTangent) {
    auto& ctx = *verts.pos.get().pCtx();
    auto& env = ctx.env();
    auto preferredQueueType = ctx.isQueueValid(zs::vk_queue_e::dedicated_transfer)
                                  ? zs::vk_queue_e::dedicated_transfer
                                  : zs::vk_queue_e::transfer;
    auto& pool = env.pools(preferredQueueType);
    auto cmd = ctx.createCommandBuffer(vk_cmd_usage_e::single_use, preferredQueueType, false);
    auto copyQueue = ctx.getLastQueue(preferredQueueType);

    cmd.begin(vk::CommandBufferBeginInfo{});

    updateAttribsFromMesh(*cmd, surfs, updatePos, updateColor, updateUv, updateNormal,
                          updateTangent);

    cmd.end();
    auto& fence = *pool.fence;

    ctx.device.resetFences({fence}, ctx.dispatcher);
    vk::CommandBuffer cmd_ = *cmd;
    auto submitInfo = vk::SubmitInfo().setCommandBufferCount(1).setPCommandBuffers(&cmd_);
    auto res = copyQueue.submit(1, &submitInfo, fence, ctx.dispatcher);
    fence.wait();
  }
  void VkModel::updateAttribsFromMesh(vk::CommandBuffer cmd, const Mesh<float, 3, u32, 3>& surfs,
                                      bool updatePos, bool updateColor, bool updateUv,
                                      bool updateNormal, bool updateTangent) {
    using Ti = u32;

    const auto& vs = surfs.nodes;
    const auto& clrs = surfs.colors;
    const auto& uvs = surfs.uvs;
    const auto& nrms = surfs.norms;
    const auto& tans = surfs.tans;

    if (verts.vertexCount == 0 || !verts.pos
        || !(updatePos || updateColor || updateUv || updateNormal || updateTangent))
      return;
    if (verts.vertexCount != vs.size()) {
      std::cerr << "the zs mesh to update from has a different point size " << vs.size()
                << " with current size " << verts.vertexCount << std::endl;
      return;
    }
    // indexCount = is.size() * 3;

    vk::BufferCopy copyRegion{};

    vk::DeviceSize numBytes;
    /// @note pos
    if (updatePos) {
      numBytes = sizeof(float) * 3 * vs.size();
      memcpy(stagingBuffer.get().mappedAddress(), vs.data(), numBytes);
      copyRegion.size = numBytes;
      cmd.copyBuffer(stagingBuffer.get(), verts.pos.get(), {copyRegion});
    }

    /// @note colors
    if (updateColor) {
      assert(verts.vertexCount == clrs.size());
      numBytes = sizeof(float) * 3 * clrs.size();
      memcpy(stagingColorBuffer.get().mappedAddress(), clrs.data(), numBytes);
      copyRegion.size = numBytes;
      cmd.copyBuffer(stagingColorBuffer.get(), verts.clr.get(), {copyRegion});
    }

    /// @note normals
    if (updateNormal || updatePos) {
      numBytes = sizeof(float) * 3 * verts.vertexCount;
      if (verts.vertexCount == nrms.size()) {
        memcpy(stagingNrmBuffer.get().mappedAddress(), nrms.data(), numBytes);
      } else {
        std::vector<std::array<float, 3>> vals(verts.vertexCount);
        compute_mesh_normal(surfs, 1.f, vals);
        memcpy(stagingNrmBuffer.get().mappedAddress(), vals.data(), numBytes);
      }
      copyRegion.size = numBytes;
      cmd.copyBuffer(stagingNrmBuffer.get(), verts.nrm.get(), {copyRegion});
    }

    /// @note tangent
    if (updateTangent) {
      assert(verts.vertexCount == tans.size());
      numBytes = sizeof(float) * 3 * verts.vertexCount;
      memcpy(stagingTangentBuffer.get().mappedAddress(), tans.data(), numBytes);
      copyRegion.size = numBytes;
      cmd.copyBuffer(stagingTangentBuffer.get(), verts.tan.get(), {copyRegion});
    }

    /// @note uvs
    if (updateUv) {
      assert(verts.vertexCount == uvs.size());
      numBytes = sizeof(float) * 2 * uvs.size();
      memcpy(stagingUVBuffer.get().mappedAddress(), uvs.data(), numBytes);
      copyRegion.size = numBytes;
      cmd.copyBuffer(stagingUVBuffer.get(), verts.uv.get(), {copyRegion});
    }
  }

  void VkModel::updatePointTextureId(const int* texIds, size_t numBytes) {
    auto& ctx = *verts.pos.get().pCtx();
    auto& env = ctx.env();
    auto preferredQueueType = ctx.isQueueValid(zs::vk_queue_e::dedicated_transfer)
                                  ? zs::vk_queue_e::dedicated_transfer
                                  : zs::vk_queue_e::transfer;
    auto& pool = env.pools(preferredQueueType);
    auto cmd = ctx.createCommandBuffer(vk_cmd_usage_e::single_use, preferredQueueType, false);
    auto copyQueue = ctx.getLastQueue(preferredQueueType);

    cmd.begin(vk::CommandBufferBeginInfo{});

    updatePointTextureId(*cmd, texIds, numBytes);

    cmd.end();
    auto& fence = *pool.fence;

    ctx.device.resetFences({fence}, ctx.dispatcher);
    vk::CommandBuffer cmd_ = *cmd;
    auto submitInfo = vk::SubmitInfo().setCommandBufferCount(1).setPCommandBuffers(&cmd_);
    auto res = copyQueue.submit(1, &submitInfo, fence, ctx.dispatcher);
    fence.wait();
  }

  void VkModel::updatePointTextureId(vk::CommandBuffer cmd, const int* texIds, size_t numBytes) {
    if (verts.vertexCount == 0 || !verts.pos || (texIds && numBytes == 0)) return;

    if (texIds == nullptr) numBytes = verts.vertexCount * 2 * sizeof(int);
    /// @note texture ids
    auto& ctx = *verts.pos.get().pCtx();
    if (!stagingTexIdBuffer || numBytes > stagingTexIdBuffer.get().getSize()) {
      if (stagingTexIdBuffer) stagingTexIdBuffer.get().unmap();
      stagingTexIdBuffer = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
      stagingTexIdBuffer.get().map();
    }
    if (!verts.texid || numBytes > verts.texid.get().getSize())
      verts.texid = ctx.createBuffer(
          numBytes, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
          vk::MemoryPropertyFlagBits::eDeviceLocal);

    if (texIds)
      memcpy(stagingTexIdBuffer.get().mappedAddress(), texIds, numBytes);
    else {
      std::vector<int> tmp(verts.vertexCount * 2, 0);
      memcpy(stagingTexIdBuffer.get().mappedAddress(), tmp.data(), numBytes);
    }

    vk::BufferCopy copyRegion{};
    copyRegion.size = numBytes;
    cmd.copyBuffer(stagingTexIdBuffer.get(), verts.texid.get(), {copyRegion});
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

  std::vector<vk::VertexInputBindingDescription> VkModel::get_binding_descriptions(
      draw_category_e e) noexcept {
    switch (e) {
      case draw_category_e::tri:
        return std::vector<vk::VertexInputBindingDescription>{
            {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {1, /*normal*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {2, /*color*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {3, /*uv*/ sizeof(float) * 2, vk::VertexInputRate::eVertex}};
      case draw_category_e::line:
        return std::vector<vk::VertexInputBindingDescription>{
            {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {1, /*color*/ sizeof(float) * 3, vk::VertexInputRate::eVertex}};
      case draw_category_e::point:
        // radius is specified through push constant
        return std::vector<vk::VertexInputBindingDescription>{
            {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {1, /*color*/ sizeof(float) * 3, vk::VertexInputRate::eVertex}};
      case draw_category_e::pick:
        return std::vector<vk::VertexInputBindingDescription>{
            {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {1, /*vid*/ sizeof(u32), vk::VertexInputRate::eVertex}};
      case draw_category_e::normal:
        return {
            {0, /* pos */ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {1, /* normal */ sizeof(float) * 3, vk::VertexInputRate::eVertex},
        };
      default:;
    }
    return {};
  }
  std::vector<vk::VertexInputAttributeDescription> VkModel::get_attribute_descriptions(
      draw_category_e e) noexcept {
    switch (e) {
      case draw_category_e::tri:
        return std::vector<vk::VertexInputAttributeDescription>{
            {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 2, /*binding*/ 2, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 3, /*binding*/ 3, vk::Format::eR32G32Sfloat, /*offset*/ (u32)0}};
      case draw_category_e::line:
        return std::vector<vk::VertexInputAttributeDescription>{
            {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0}};
      case draw_category_e::point:
        return std::vector<vk::VertexInputAttributeDescription>{
            {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0}};
      case draw_category_e::pick:
        return std::vector<vk::VertexInputAttributeDescription>{
            {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 1, /*binding*/ 1, vk::Format::eR32Uint, /*offset*/ (u32)0}};
      case draw_category_e::normal:
        return {{/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
                {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0}};
      default:;
    }
    return {};
  }
  void VkModel::draw(const vk::CommandBuffer& cmd, draw_category_e e) const {
    switch (e) {
      case draw_category_e::tri:
        cmd.drawIndexed(/*index count*/ indexCount, /*instance count*/ 1,
                        /*first index*/ 0, /*vertex offset*/ 0,
                        /*first instance*/ 0, indices.get().ctx.dispatcher);
        break;
      case draw_category_e::line:
        cmd.drawIndexed(/*index count*/ indexCount, /*instance count*/ 1,
                        /*first index*/ 0, /*vertex offset*/ 0,
                        /*first instance*/ 0, indices.get().ctx.dispatcher);
        break;
      case draw_category_e::point:
      case draw_category_e::pick:
      case draw_category_e::normal:
        cmd.draw(/*vertex count*/ verts.vertexCount, /*instance count*/ 1,
                 /*first vertex*/ 0, /*first instance*/ 0, indices.get().ctx.dispatcher);
        break;
      default:;
    }
  }

  void VkModel::bind(const vk::CommandBuffer& cmd, draw_category_e e) const {
    switch (e) {
      case draw_category_e::tri: {
        std::array<vk::Buffer, 4> buffers{verts.pos.get(), verts.nrm.get(), verts.clr.get(),
                                          verts.uv.get()};
        std::array<vk::DeviceSize, 4> offsets{0, 0, 0, 0};
        cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets, verts.pos.get().ctx.dispatcher);
        cmd.bindIndexBuffer((vk::Buffer)indices.get(), /*offset*/ (u32)0, vk::IndexType::eUint32,
                            indices.get().ctx.dispatcher);
      } break;
      case draw_category_e::line: {
        std::array<vk::Buffer, 2> buffers{verts.pos.get(), verts.clr.get()};
        std::array<vk::DeviceSize, 2> offsets{0, 0};
        cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets, verts.pos.get().ctx.dispatcher);
        cmd.bindIndexBuffer((vk::Buffer)indices.get(), /*offset*/ (u32)0, vk::IndexType::eUint32,
                            indices.get().ctx.dispatcher);
      } break;
      case draw_category_e::point: {
        std::array<vk::Buffer, 2> buffers{verts.pos.get(), verts.clr.get()};
        std::array<vk::DeviceSize, 2> offsets{0, 0};
        cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets, verts.pos.get().ctx.dispatcher);
      } break;
      case draw_category_e::pick: {
        std::array<vk::Buffer, 2> buffers{verts.pos.get(), verts.vids.get()};
        std::array<vk::DeviceSize, 2> offsets{0, 0};
        cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets, verts.pos.get().ctx.dispatcher);
      } break;
      case draw_category_e::normal: {
        std::array<vk::Buffer, 2> buffers{verts.pos.get(), verts.nrm.get()};
        std::array<vk::DeviceSize, 2> offsets{0, 0};
        cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets, verts.pos.get().ctx.dispatcher);
      } break;
      default:;
    }
  }

  std::vector<vk::VertexInputBindingDescription> VkModel::get_binding_descriptions_color(
      draw_category_e e) noexcept {
    switch (e) {
      case draw_category_e::tri:
        return std::vector<vk::VertexInputBindingDescription>{
            {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {1, /*color*/ sizeof(float) * 3, vk::VertexInputRate::eVertex}};
      case draw_category_e::line:
        return std::vector<vk::VertexInputBindingDescription>{
            {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {1, /*color*/ sizeof(float) * 3, vk::VertexInputRate::eVertex}};
      case draw_category_e::point:
        // radius is specified through push constant
        return std::vector<vk::VertexInputBindingDescription>{
            {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {1, /*color*/ sizeof(float) * 3, vk::VertexInputRate::eVertex}};
      default:;
    }
    return {};
  }
  std::vector<vk::VertexInputAttributeDescription> VkModel::get_attribute_descriptions_color(
      draw_category_e e) noexcept {
    switch (e) {
      case draw_category_e::tri:
        return std::vector<vk::VertexInputAttributeDescription>{
            {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0}};
      case draw_category_e::line:
        return std::vector<vk::VertexInputAttributeDescription>{
            {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0}};
      case draw_category_e::point:
        return std::vector<vk::VertexInputAttributeDescription>{
            {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0}};
      default:;
    }
    return {};
  }
  void VkModel::drawColor(const vk::CommandBuffer& cmd, draw_category_e e) const {
    switch (e) {
      case draw_category_e::tri:
        cmd.drawIndexed(/*index count*/ indexCount, /*instance count*/ 1,
                        /*first index*/ 0, /*vertex offset*/ 0,
                        /*first instance*/ 0, indices.get().ctx.dispatcher);
        break;
      case draw_category_e::line:
        cmd.drawIndexed(/*index count*/ indexCount, /*instance count*/ 1,
                        /*first index*/ 0, /*vertex offset*/ 0,
                        /*first instance*/ 0, indices.get().ctx.dispatcher);
        break;
      case draw_category_e::point:
        cmd.draw(/*vertex count*/ verts.vertexCount, /*instance count*/ 1,
                 /*first vertex*/ 0, /*first instance*/ 0, indices.get().ctx.dispatcher);
        break;
      default:;
    }
  }

  void VkModel::bindColor(const vk::CommandBuffer& cmd, draw_category_e e) const {
    switch (e) {
      case draw_category_e::tri: {
        std::array<vk::Buffer, 2> buffers{verts.pos.get(), verts.clr.get()};
        std::array<vk::DeviceSize, 2> offsets{0, 0};
        cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets, verts.pos.get().ctx.dispatcher);
        cmd.bindIndexBuffer((vk::Buffer)indices.get(), /*offset*/ (u32)0, vk::IndexType::eUint32,
                            indices.get().ctx.dispatcher);
      } break;
      case draw_category_e::line: {
        std::array<vk::Buffer, 2> buffers{verts.pos.get(), verts.clr.get()};
        std::array<vk::DeviceSize, 2> offsets{0, 0};
        cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets, verts.pos.get().ctx.dispatcher);
        cmd.bindIndexBuffer((vk::Buffer)indices.get(), /*offset*/ (u32)0, vk::IndexType::eUint32,
                            indices.get().ctx.dispatcher);
      } break;
      case draw_category_e::point: {
        std::array<vk::Buffer, 2> buffers{verts.pos.get(), verts.clr.get()};
        std::array<vk::DeviceSize, 2> offsets{0, 0};
        cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets, verts.pos.get().ctx.dispatcher);
      } break;
      default:;
    }
  }

  std::vector<vk::VertexInputBindingDescription> VkModel::get_binding_descriptions_normal_color(
      draw_category_e e) noexcept {
    switch (e) {
      case draw_category_e::tri:
        return std::vector<vk::VertexInputBindingDescription>{
            {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {1, /*normal*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {2, /*color*/ sizeof(float) * 3, vk::VertexInputRate::eVertex}};
      case draw_category_e::line:
        return std::vector<vk::VertexInputBindingDescription>{
            {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {1, /*color*/ sizeof(float) * 3, vk::VertexInputRate::eVertex}};
      case draw_category_e::point:
        // radius is specified through push constant
        return std::vector<vk::VertexInputBindingDescription>{
            {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {1, /*color*/ sizeof(float) * 3, vk::VertexInputRate::eVertex}};
      default:;
    }
    return {};
  }
  std::vector<vk::VertexInputAttributeDescription> VkModel::get_attribute_descriptions_normal_color(
      draw_category_e e) noexcept {
    switch (e) {
      case draw_category_e::tri:
        return std::vector<vk::VertexInputAttributeDescription>{
            {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 2, /*binding*/ 2, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0}};
      case draw_category_e::line:
        return std::vector<vk::VertexInputAttributeDescription>{
            {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0}};
      case draw_category_e::point:
        return std::vector<vk::VertexInputAttributeDescription>{
            {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0}};
      default:;
    }
    return {};
  }
  void VkModel::drawNormalColor(const vk::CommandBuffer& cmd, draw_category_e e) const {
    switch (e) {
      case draw_category_e::tri:
        cmd.drawIndexed(/*index count*/ indexCount, /*instance count*/ 1,
                        /*first index*/ 0, /*vertex offset*/ 0,
                        /*first instance*/ 0, indices.get().ctx.dispatcher);
        break;
      case draw_category_e::line:
        cmd.drawIndexed(/*index count*/ indexCount, /*instance count*/ 1,
                        /*first index*/ 0, /*vertex offset*/ 0,
                        /*first instance*/ 0, indices.get().ctx.dispatcher);
        break;
      case draw_category_e::point:
        cmd.draw(/*vertex count*/ verts.vertexCount, /*instance count*/ 1,
                 /*first vertex*/ 0, /*first instance*/ 0, indices.get().ctx.dispatcher);
        break;
      default:;
    }
  }

  void VkModel::bindNormalColor(const vk::CommandBuffer& cmd, draw_category_e e) const {
    switch (e) {
      case draw_category_e::tri: {
        std::array<vk::Buffer, 3> buffers{verts.pos.get(), verts.nrm.get(), verts.clr.get()};
        std::array<vk::DeviceSize, 3> offsets{0, 0, 0};
        cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets, verts.pos.get().ctx.dispatcher);
        cmd.bindIndexBuffer((vk::Buffer)indices.get(), /*offset*/ (u32)0, vk::IndexType::eUint32,
                            indices.get().ctx.dispatcher);
      } break;
      case draw_category_e::line: {
        std::array<vk::Buffer, 2> buffers{verts.pos.get(), verts.clr.get()};
        std::array<vk::DeviceSize, 2> offsets{0, 0};
        cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets, verts.pos.get().ctx.dispatcher);
        cmd.bindIndexBuffer((vk::Buffer)indices.get(), /*offset*/ (u32)0, vk::IndexType::eUint32,
                            indices.get().ctx.dispatcher);
      } break;
      case draw_category_e::point: {
        std::array<vk::Buffer, 2> buffers{verts.pos.get(), verts.clr.get()};
        std::array<vk::DeviceSize, 2> offsets{0, 0};
        cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets, verts.pos.get().ctx.dispatcher);
      } break;
      default:;
    }
  }

  std::vector<vk::VertexInputBindingDescription> VkModel::get_binding_descriptions_uv(
      draw_category_e e) noexcept {
    switch (e) {
      case draw_category_e::tri:
        return std::vector<vk::VertexInputBindingDescription>{
            {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {1, /*normal*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {2, /*uv*/ sizeof(float) * 2, vk::VertexInputRate::eVertex},
            {3, /*texid*/ sizeof(int) * 2, vk::VertexInputRate::eVertex},
        };
      case draw_category_e::line:
        return std::vector<vk::VertexInputBindingDescription>{
            {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {1, /*color*/ sizeof(float) * 3, vk::VertexInputRate::eVertex}};
      case draw_category_e::point:
        // radius is specified through push constant
        return std::vector<vk::VertexInputBindingDescription>{
            {0, /*pos*/ sizeof(float) * 3, vk::VertexInputRate::eVertex},
            {1, /*color*/ sizeof(float) * 3, vk::VertexInputRate::eVertex}};
      default:;
    }
    return {};
  }
  std::vector<vk::VertexInputAttributeDescription> VkModel::get_attribute_descriptions_uv(
      draw_category_e e) noexcept {
    switch (e) {
      case draw_category_e::tri:
        return std::vector<vk::VertexInputAttributeDescription>{
            {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 2, /*binding*/ 2, vk::Format::eR32G32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 3, /*binding*/ 3, vk::Format::eR32G32Sint, /*offset*/ (u32)0},
        };
      case draw_category_e::line:
        return std::vector<vk::VertexInputAttributeDescription>{
            {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0}};
      case draw_category_e::point:
        return std::vector<vk::VertexInputAttributeDescription>{
            {/*location*/ 0, /*binding*/ 0, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0},
            {/*location*/ 1, /*binding*/ 1, vk::Format::eR32G32B32Sfloat, /*offset*/ (u32)0}};
      default:;
    }
    return {};
  }
  void VkModel::drawUV(const vk::CommandBuffer& cmd, draw_category_e e) const {
    switch (e) {
      case draw_category_e::tri:
        cmd.drawIndexed(/*index count*/ indexCount, /*instance count*/ 1,
                        /*first index*/ 0, /*vertex offset*/ 0,
                        /*first instance*/ 0, indices.get().ctx.dispatcher);
        break;
      case draw_category_e::line:
        cmd.drawIndexed(/*index count*/ indexCount, /*instance count*/ 1,
                        /*first index*/ 0, /*vertex offset*/ 0,
                        /*first instance*/ 0, indices.get().ctx.dispatcher);
        break;
      case draw_category_e::point:
        cmd.draw(/*vertex count*/ verts.vertexCount, /*instance count*/ 1,
                 /*first vertex*/ 0, /*first instance*/ 0, indices.get().ctx.dispatcher);
        break;
      default:;
    }
  }

  void VkModel::bindUV(const vk::CommandBuffer& cmd, draw_category_e e) const {
    switch (e) {
      case draw_category_e::tri: {
        std::array<vk::Buffer, 4> buffers{verts.pos.get(), verts.nrm.get(), verts.uv.get(),
                                          verts.texid.get()};
        std::array<vk::DeviceSize, 4> offsets{0, 0, 0, 0};
        cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets, verts.pos.get().ctx.dispatcher);
        cmd.bindIndexBuffer((vk::Buffer)indices.get(), /*offset*/ (u32)0, vk::IndexType::eUint32,
                            indices.get().ctx.dispatcher);
      } break;
      case draw_category_e::line: {
        std::array<vk::Buffer, 2> buffers{verts.pos.get(), verts.clr.get()};
        std::array<vk::DeviceSize, 2> offsets{0, 0};
        cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets, verts.pos.get().ctx.dispatcher);
        cmd.bindIndexBuffer((vk::Buffer)indices.get(), /*offset*/ (u32)0, vk::IndexType::eUint32,
                            indices.get().ctx.dispatcher);
      } break;
      case draw_category_e::point: {
        std::array<vk::Buffer, 2> buffers{verts.pos.get(), verts.clr.get()};
        std::array<vk::DeviceSize, 2> offsets{0, 0};
        cmd.bindVertexBuffers(/*firstBinding*/ 0, buffers, offsets, verts.pos.get().ctx.dispatcher);
      } break;
      default:;
    }
  }

}  // namespace zs