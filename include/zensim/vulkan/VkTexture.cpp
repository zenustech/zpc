#include "zensim/vulkan/VkTexture.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "zensim/zpc_tpls/stb/stb_image.h"

namespace zs {

  VkTexture load_texture(VulkanContext &ctx, u8 *data, size_t numBytes, vk::Extent2D extent,
                         vk::Format format, vk::ImageLayout layout) {
    VkTexture ret;
    /// attribs
    ret.imageLayout = layout;
    /// sampler
    auto samplerCI = vk::SamplerCreateInfo{}
                         .setMaxAnisotropy(1.f)
                         .setMagFilter(vk::Filter::eLinear)
                         .setMinFilter(vk::Filter::eLinear)
                         .setMipmapMode(vk::SamplerMipmapMode::eLinear)
                         .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
                         .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
                         .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
                         .setBorderColor(vk::BorderColor::eFloatOpaqueWhite);
    ret.sampler = ctx.device.createSampler(samplerCI, nullptr, ctx.dispatcher);
    /// image
    auto stagingBuffer = ctx.createStagingBuffer(numBytes, vk::BufferUsageFlagBits::eTransferSrc);
    stagingBuffer.map();
    memcpy(stagingBuffer.mappedAddress(), data, numBytes);
    stagingBuffer.unmap();

    auto img = ctx.createOptimal2DImage(
        extent, format, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        /*MemoryPropertyFlags*/ vk::MemoryPropertyFlagBits::eDeviceLocal,
        /*mipmaps*/ false,
        /*createView*/ true);

    // transition
    auto &env = ctx.env();
    auto &pool = env.pools(zs::vk_queue_e::graphics);
    // auto copyQueue = env.pools(zs::vk_queue_e::transfer).queue;
    auto copyQueue = pool.queue;
    vk::CommandBuffer cmd = pool.createCommandBuffer(vk::CommandBufferLevel::ePrimary, false,
                                                     nullptr, zs::vk_cmd_usage_e::single_use);
    {
      cmd.begin(vk::CommandBufferBeginInfo{});

      // set image layout
      auto imageBarrier = image_layout_transition_barrier(
          img, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined,
          vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eHost,
          vk::PipelineStageFlagBits::eTransfer);
      cmd.pipelineBarrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer,
                          vk::DependencyFlags(), {}, {}, {imageBarrier}, ctx.dispatcher);

      // Copy
      auto bufferCopyRegion = vk::BufferImageCopy{};
      bufferCopyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
      bufferCopyRegion.imageSubresource.layerCount = 1;
      bufferCopyRegion.imageExtent = vk::Extent3D{(u32)extent.width, (u32)extent.height, (u32)1};

      cmd.copyBufferToImage(stagingBuffer, img, vk::ImageLayout::eTransferDstOptimal,
                            {bufferCopyRegion}, ctx.dispatcher);

      // set image layout
      imageBarrier = image_layout_transition_barrier(
          img, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal, layout,
          vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader);
      cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                          vk::PipelineStageFlagBits::eFragmentShader, vk::DependencyFlags(), {}, {},
                          {imageBarrier}, ctx.dispatcher);
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
    }
    ctx.device.freeCommandBuffers(pool.cmdpool(zs::vk_cmd_usage_e::single_use), cmd,
                                  ctx.dispatcher);

    // ret.image = std::make_unique<Image>(std::move(img));
    ret.image = std::move(img);
    return ret;
  }

}  // namespace zs