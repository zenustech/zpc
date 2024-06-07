#include "zensim/vulkan/VkSwapchain.hpp"

#include "zensim/Logger.hpp"
#include "zensim/types/Iterator.h"
#include "zensim/vulkan/VkImage.hpp"
#include "zensim/vulkan/VkRenderPass.hpp"
#include "zensim/vulkan/Vulkan.hpp"

namespace zs {

  ///
  ///
  /// swapchain builder
  ///
  ///
  vk::Result Swapchain::acquireNextImage(u32& imageId) {
    if (vk::Result res = ctx.device.waitForFences(
            1, &currentFence(), VK_TRUE, detail::deduce_numeric_max<u64>(), ctx.dispatcher);
        res != vk::Result::eSuccess)
      throw std::runtime_error(fmt::format(
          "[acquireNextImage]: Failed to wait for fence at frame [{}] with result [{}]\n",
          frameIndex, res));
#if 0
    auto res = ctx.device.acquireNextImageKHR(
        swapchain, detail::deduce_numeric_max<u64>(),
        currentImageAcquiredSemaphore(),  // must be a not signaled semaphore
        VK_NULL_HANDLE, ctx.dispatcher);
    imageId = res.value;
    return res.result;
#else
    auto result = ctx.dispatcher.vkAcquireNextImageKHR(
        (vk::Device)ctx.device, (vk::SwapchainKHR)swapchain, detail::deduce_numeric_max<u64>(),
        currentImageAcquiredSemaphore(), VK_NULL_HANDLE, &imageId);
    return vk::Result{result};
#endif
#if 0
    if (res.result != vk::Result::eSuccess)
      throw std::runtime_error(fmt::format(
          "[acquireNextImage]: Failed to acquire next image at frame [{}] with result [{}]\n",
          frameIndex, res.result));
    return res.value;
#endif
  }
  vk::Result Swapchain::present(vk::Queue queue, u32 imageId) {
    vk::SwapchainKHR swapChains[] = {swapchain};
    vk::Result presentResult = vk::Result::eSuccess;
    auto presentInfo = vk::PresentInfoKHR{}
                           .setSwapchainCount(1)
                           .setPSwapchains(swapChains)
                           .setPImageIndices(&imageId)
                           .setPResults(&presentResult)
                           .setWaitSemaphoreCount(1)
                           .setPWaitSemaphores(&currentRenderCompleteSemaphore());

    // https://github.com/KhronosGroup/Vulkan-Hpp/issues/599
    return static_cast<vk::Result>(ctx.dispatcher.vkQueuePresentKHR(
        static_cast<VkQueue>(queue), reinterpret_cast<const VkPresentInfoKHR*>(&presentInfo)));
  }

  RenderPass Swapchain::getRenderPass() {
    const bool enableMS = multiSampleEnabled();
    const bool enableDepth = depthEnabled();

    auto rpBuilder = ctx.renderpass().setNumPasses(1);
    if (enableMS)
      rpBuilder.addAttachment(ci.imageFormat, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eColorAttachmentOptimal, true, sampleBits);
    rpBuilder.addAttachment(ci.imageFormat, vk::ImageLayout::eUndefined,
                            vk::ImageLayout::ePresentSrcKHR, !enableMS);
    if (enableDepth)
      rpBuilder.addAttachment(depthFormat, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eDepthStencilAttachmentOptimal, true, sampleBits);
    if (enableMS) {
      vk::AccessFlags accessMask = vk::AccessFlagBits::eColorAttachmentWrite;
      if (enableDepth) accessMask = accessMask | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
      rpBuilder
          .addSubpass(/*color*/ {0}, /*ds ref*/ enableDepth ? 2 : -1,
                      /*color resolve ref*/ {1}) /*input*/
          .setSubpassDependencies(
              {vk::SubpassDependency2{}
                   .setSrcSubpass(VK_SUBPASS_EXTERNAL)
                   .setDstSubpass(0)
                   .setSrcAccessMask(accessMask)
                   .setDstAccessMask(accessMask)
                   .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput
                                    | vk::PipelineStageFlagBits::eLateFragmentTests)
                   .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput
                                    | vk::PipelineStageFlagBits::eEarlyFragmentTests)});
    }
    return rpBuilder.build();
  }
  /// @note default framebuffer setup
  void Swapchain::initFramebuffersFor(vk::RenderPass renderPass) {
    frameBuffers.clear();
    auto cnt = imageCount();
    const bool enableMS = multiSampleEnabled();
    const bool enableDepth = depthEnabled();
    std::vector<vk::ImageView> imgvs;
    imgvs.reserve(3);
    for (int i = 0; i != cnt; ++i) {
      imgvs.clear();
      if (enableMS) imgvs.push_back((vk::ImageView)msColorBuffers[i]);
      imgvs.push_back((vk::ImageView)imageViews[i]);
      if (enableDepth) imgvs.push_back((vk::ImageView)depthBuffers[i]);

      frameBuffers.emplace_back(ctx.createFramebuffer(imgvs, ci.imageExtent, renderPass));
    }
  }

  SwapchainBuilder::SwapchainBuilder(VulkanContext& ctx, vk::SurfaceKHR targetSurface)
      : ctx{ctx}, surface{targetSurface} {
    ZS_ERROR_IF(!ctx.supportSurface(surface),
                fmt::format("queue [{}] does not support this surface!\n",
                            ctx.queueFamilyIndices[vk_queue_e::graphics]));
    surfFormats = ctx.physicalDevice.getSurfaceFormatsKHR(surface, ctx.dispatcher);
    surfCapabilities = ctx.physicalDevice.getSurfaceCapabilitiesKHR(surface, ctx.dispatcher);
    surfPresentModes = ctx.physicalDevice.getSurfacePresentModesKHR(surface, ctx.dispatcher);
    swapchainDepthFormat = ctx.findSupportedFormat(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal, vk::FormatFeatureFlagBits::eDepthStencilAttachment);
    sampleBits = vk::SampleCountFlagBits::e1;

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
  void SwapchainBuilder::resize(Swapchain& obj, u32 width, u32 height) {
    /// @note credits
    /// https://www.reddit.com/r/vulkan/comments/cc3edr/swapchain_recreation_repeatedly_returns_vk_error/
    surfCapabilities = ctx.physicalDevice.getSurfaceCapabilitiesKHR(obj.ci.surface, ctx.dispatcher);
    width = std::clamp(width, surfCapabilities.minImageExtent.width,
                       surfCapabilities.maxImageExtent.width);
    height = std::clamp(height, surfCapabilities.minImageExtent.height,
                        surfCapabilities.maxImageExtent.height);
    obj.ci.imageExtent = vk::Extent2D{width, height};

    build(obj);
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
    // kept the previously built swapchain for this
    const bool rebuild = obj.swapchain != VK_NULL_HANDLE;
    if (rebuild) ci = obj.ci;
    ci.oldSwapchain = obj.swapchain;
    obj.swapchain = ctx.device.createSwapchainKHR(ci, nullptr, ctx.dispatcher);

    obj.frameIndex = 0;

#if 0
    obj.extent = ci.imageExtent;
    obj.colorFormat = ci.imageFormat;
    obj.imageColorSpace = ci.imageColorSpace;
    obj.depthFormat = swapchainDepthFormat;
    obj.presentMode = ci.presentMode;
#else
    if (!rebuild) {
      obj.depthFormat = swapchainDepthFormat;
      obj.sampleBits = sampleBits;
    }
    obj.ci = ci;
    // if (obj.swapchain == VK_NULL_HANDLE)
    //   obj.ci = ci;
    // else
    //   obj.ci.oldSwapchain = ci.oldSwapchain;
#endif

    obj.resetAux();
    obj.images = ctx.device.getSwapchainImagesKHR(obj.swapchain, ctx.dispatcher);
    const auto numSwapchainImages = obj.images.size();

    /// reset previous resources (if any)
    ctx.device.destroySwapchainKHR(ci.oldSwapchain, nullptr, ctx.dispatcher);

    /// construct current swapchain
    obj.imageViews.resize(numSwapchainImages);
    for (int i = 0; i != numSwapchainImages; ++i) {
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
    if (obj.sampleBits != vk::SampleCountFlagBits::e1) {
      for (int i = 0; i != numSwapchainImages; ++i) {
        obj.msColorBuffers.emplace_back(ctx.create2DImage(
            ci.imageExtent, ci.imageFormat,
            vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal, /*mipmaps*/ false, /*createView*/ true,
            /*enable transfer*/ false, obj.sampleBits));
      }
    }
    if (buildDepthBuffer) {
      for (int i = 0; i != numSwapchainImages; ++i) {
        obj.depthBuffers.emplace_back(ctx.create2DImage(
            ci.imageExtent, obj.depthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal, /*mipmaps*/ false, /*createView*/ true,
            /*enable transfer*/ false, obj.sampleBits));
      }
    }

    /// sync primitives
    obj.imageFences.resize(numSwapchainImages, VK_NULL_HANDLE);
    if (obj.imageAcquiredSemaphores.size() != num_buffered_frames) {
      obj.imageAcquiredSemaphores.resize(num_buffered_frames);
      obj.renderCompleteSemaphores.resize(num_buffered_frames);
      obj.fences.resize(num_buffered_frames);
      for (int i = 0; i != num_buffered_frames; ++i) {
        // semaphores
        obj.imageAcquiredSemaphores[i]
            = ctx.device.createSemaphore(vk::SemaphoreCreateInfo{}, nullptr, ctx.dispatcher);
        obj.renderCompleteSemaphores[i]
            = ctx.device.createSemaphore(vk::SemaphoreCreateInfo{}, nullptr, ctx.dispatcher);
        obj.fences[i] = ctx.device.createFence(
            vk::FenceCreateInfo{}.setFlags(vk::FenceCreateFlagBits::eSignaled), nullptr,
            ctx.dispatcher);
      }
    }
  }

}  // namespace zs