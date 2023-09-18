#include "zensim/vulkan/VkSwapchain.hpp"

#include "zensim/Logger.hpp"
#include "zensim/types/Iterator.h"
#include "zensim/vulkan/VkImage.hpp"
#include "zensim/vulkan/VkRenderPass.hpp"

namespace zs {

  ///
  ///
  /// swapchain builder
  ///
  ///
  u32 Swapchain::acquireNextImage() {
    if (vk::Result res = ctx.device.waitForFences(
            1, &readFences[frameIndex], VK_TRUE, detail::deduce_numeric_max<u64>(), ctx.dispatcher);
        res != vk::Result::eSuccess)
      throw std::runtime_error(fmt::format(
          "[acquireNextImage]: Failed to wait for fence at frame [{}] with result [{}]\n",
          frameIndex, res));
    auto res = ctx.device.acquireNextImageKHR(
        swapchain, detail::deduce_numeric_max<u64>(),
        readSemaphores[frameIndex],  // must be a not signaled semaphore
        VK_NULL_HANDLE, ctx.dispatcher);
    if (res.result != vk::Result::eSuccess)
      throw std::runtime_error(fmt::format(
          "[acquireNextImage]: Failed to acquire next image at frame [{}] with result [{}]\n",
          frameIndex, res.result));
    return res.value;
  }
  RenderPass Swapchain::getRenderPass() {
    RenderPass ret{ctx};
    const bool includeDepthBuffer = depthBuffers.size() == imageCount() && depthBuffers.size() != 0;

    // fmt::print("\n\ncheck depthbuffers: {} (image count {})\n\n", depthBuffers.size(),
    //           imageCount());
    std::vector<vk::AttachmentDescription> attachments(1 + (includeDepthBuffer ? 1 : 0));
    std::vector<vk::AttachmentReference> refs(attachments.size());
    // color
    auto& colorAttachment = attachments[0];
    colorAttachment = vk::AttachmentDescription{}
                          .setFormat(colorFormat)
                          .setSamples(vk::SampleCountFlagBits::e1)
                          .setLoadOp(vk::AttachmentLoadOp::eClear)
                          .setStoreOp(vk::AttachmentStoreOp::eStore)
                          .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                          .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                          .setInitialLayout(vk::ImageLayout::eUndefined)
                          .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

    auto& colorAttachmentRef = refs[0];
    colorAttachmentRef = vk::AttachmentReference{0, vk::ImageLayout::eColorAttachmentOptimal};

    // depth
    if (includeDepthBuffer) {
      auto& depthAttachment = attachments[1];
      depthAttachment = vk::AttachmentDescription{}
                            .setFormat(depthFormat)
                            .setSamples(vk::SampleCountFlagBits::e1)
                            .setLoadOp(vk::AttachmentLoadOp::eClear)
                            .setStoreOp(vk::AttachmentStoreOp::eDontCare)
                            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                            .setInitialLayout(vk::ImageLayout::eUndefined)
                            .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

      auto& depthAttachmentRef = refs[1];
      depthAttachmentRef
          = vk::AttachmentReference{1, vk::ImageLayout::eDepthStencilAttachmentOptimal};
    }

    /// @note here color attachment index corresponding to
    /// 'layout(location = k) out vec4 outColor' directive in the fragment shader
    auto subpass = vk::SubpassDescription{}
                       .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                       .setColorAttachmentCount(1)
                       .setPColorAttachments(&refs[0]);
    if (includeDepthBuffer) subpass.setPDepthStencilAttachment(&refs[1]);

    vk::AccessFlags accessFlag;
    if (includeDepthBuffer)
      accessFlag = vk::AccessFlagBits::eColorAttachmentWrite
                   | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    else
      accessFlag = vk::AccessFlagBits::eColorAttachmentWrite;
    auto dependency = vk::SubpassDependency{}
                          .setDstSubpass(0)
                          .setDstAccessMask(accessFlag)
                          .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput
                                           | vk::PipelineStageFlagBits::eEarlyFragmentTests)
                          .setSrcSubpass(VK_SUBPASS_EXTERNAL)
                          .setSrcAccessMask({})
                          .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput
                                           | vk::PipelineStageFlagBits::eEarlyFragmentTests);

    ret.renderpass = ctx.device.createRenderPass(vk::RenderPassCreateInfo{}
                                                     .setAttachmentCount(attachments.size())
                                                     .setPAttachments(attachments.data())
                                                     .setSubpassCount(1)
                                                     .setPSubpasses(&subpass)
                                                     .setDependencyCount(1)
                                                     .setPDependencies(&dependency),
                                                 nullptr, ctx.dispatcher);

    return ret;
  }
  void Swapchain::initFramebuffersFor(vk::RenderPass renderPass) {
    frameBuffers.clear();
    auto cnt = imageCount();
    if (depthBuffers.size() == cnt) {
      // color + depth
      for (int i = 0; i != cnt; ++i) {
        frameBuffers.emplace_back(ctx.createFramebuffer(
            {(vk::ImageView)imageViews[i], (vk::ImageView)depthBuffers[i]}, extent, renderPass));
      }
    } else {
      // color only
      for (int i = 0; i != imageCount(); ++i) {
        frameBuffers.emplace_back(
            ctx.createFramebuffer({(vk::ImageView)imageViews[i]}, extent, renderPass));
      }
    }
  }

  SwapchainBuilder::SwapchainBuilder(VulkanContext& ctx, vk::SurfaceKHR targetSurface)
      : ctx{ctx}, surface{targetSurface} {
    ZS_ERROR_IF(
        !ctx.supportSurface(surface),
        fmt::format("queue [{}] does not support this surface!\n", ctx.graphicsQueueFamilyIndex));
    surfFormats = ctx.physicalDevice.getSurfaceFormatsKHR(surface, ctx.dispatcher);
    surfCapabilities = ctx.physicalDevice.getSurfaceCapabilitiesKHR(surface, ctx.dispatcher);
    surfPresentModes = ctx.physicalDevice.getSurfacePresentModesKHR(surface, ctx.dispatcher);
    swapchainDepthFormat = ctx.findSupportedFormat(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal, vk::FormatFeatureFlagBits::eDepthStencilAttachment);

    ci.surface = surface;

    /// @brief default setup
    ci.minImageCount = surfCapabilities.minImageCount;
    ci.imageArrayLayers = 1;
    ci.preTransform = surfCapabilities.currentTransform;
    ci.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    ci.clipped = true;
    // ci.oldSwapchain = nullptr;

    // format and colorspace selection
    if (surfFormats.size() == 1 && surfFormats.front().format == vk::Format::eUndefined) {
      // no preferred format, select this
      ci.imageFormat = vk::Format::eB8G8R8A8Srgb;
      ci.imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
    } else {
      ci.imageFormat = surfFormats.front().format;
      ci.imageColorSpace = surfFormats.front().colorSpace;
      for (const auto& fmt : surfFormats)
        if (fmt.format == vk::Format::eB8G8R8A8Srgb
            && fmt.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
          ci.imageFormat = vk::Format::eB8G8R8A8Srgb;
          ci.imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
          break;
        }
    }

    // extent
    if (surfCapabilities.currentExtent.width == std::numeric_limits<u32>::max())
      ci.imageExtent = surfCapabilities.maxImageExtent;
    else
      ci.imageExtent = surfCapabilities.currentExtent;

    // could also be eTransferDst (other view) or eStorage (shader)
    ci.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

    // present mode selection
    ci.presentMode = vk::PresentModeKHR::eFifo;  //  required to be supported by all vendors

    ci.imageSharingMode = vk::SharingMode::eExclusive;
  }
  SwapchainBuilder& SwapchainBuilder::presentMode(vk::PresentModeKHR mode) {
    if (mode == ci.presentMode) return *this;
    for (auto& m : surfPresentModes)
      if (m == mode) {
        ci.presentMode = mode;
        return *this;
      }
    ZS_WARN(fmt::format("Present mode [{}] is not supported in this context. Ignored.\n", mode));
    return *this;
  }
  void SwapchainBuilder::build(Swapchain& obj) {
    constexpr auto num_buffered_frames = Swapchain::num_buffered_frames;
    // kept the previously built swapchain for this
    ci.oldSwapchain = obj.swapchain;
    obj.swapchain = ctx.device.createSwapchainKHR(ci, nullptr, ctx.dispatcher);
    obj.frameIndex = 0;
    obj.extent = ci.imageExtent;
    obj.colorFormat = ci.imageFormat;
    obj.depthFormat = swapchainDepthFormat;
    obj.presentMode = ci.presentMode;
    obj.imageColorSpace = ci.imageColorSpace;

    obj.resetAux();
    obj.images = ctx.device.getSwapchainImagesKHR(obj.swapchain, ctx.dispatcher);

    /// reset previous resources (if any)
    if (ci.oldSwapchain) ctx.device.destroySwapchainKHR(ci.oldSwapchain, nullptr, ctx.dispatcher);

    /// construct current swapchain
    obj.imageViews.resize(obj.images.size());
    for (int i = 0; i != obj.images.size(); ++i) {
      auto& img = obj.images[i];
      // image views
      auto subresourceRange = vk::ImageSubresourceRange()
                                  .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                  .setBaseMipLevel(0)
                                  .setLevelCount(1)
                                  .setBaseArrayLayer(0)
                                  .setLayerCount(1);
      auto ivCI = vk::ImageViewCreateInfo{{},
                                          img,
                                          vk::ImageViewType::e2D,
                                          ci.imageFormat,
                                          vk::ComponentMapping(),
                                          subresourceRange};
      obj.imageViews[i] = ctx.device.createImageView(ivCI, nullptr, ctx.dispatcher);
    }
    if (buildDepthBuffer) {
      for (int i = 0; i != obj.images.size(); ++i) {
        obj.depthBuffers.emplace_back(ctx.create2DImage(
            ci.imageExtent, swapchainDepthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal, /*mipmaps*/ false, /*createView*/ true));
      }
    }

    /// sync primitives
    if (obj.readSemaphores.size() != num_buffered_frames) {
      obj.readSemaphores.resize(num_buffered_frames);
      obj.writeSemaphores.resize(num_buffered_frames);
      obj.readFences.resize(num_buffered_frames);
      obj.writeFences.resize(num_buffered_frames);
      for (int i = 0; i != num_buffered_frames; ++i) {
        // semaphores
        obj.readSemaphores[i]
            = ctx.device.createSemaphore(vk::SemaphoreCreateInfo{}, nullptr, ctx.dispatcher);
        obj.writeSemaphores[i]
            = ctx.device.createSemaphore(vk::SemaphoreCreateInfo{}, nullptr, ctx.dispatcher);
        obj.readFences[i] = ctx.device.createFence(vk::FenceCreateInfo{}, nullptr, ctx.dispatcher);
        obj.writeFences[i] = ctx.device.createFence(vk::FenceCreateInfo{}, nullptr, ctx.dispatcher);
      }
    }
  }

}  // namespace zs