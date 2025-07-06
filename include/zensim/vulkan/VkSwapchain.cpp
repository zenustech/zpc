#include "zensim/vulkan/VkSwapchain.hpp"

#include "zensim/Logger.hpp"
#include "zensim/types/Iterator.h"
#include "zensim/vulkan/VkImage.hpp"
#include "zensim/vulkan/VkRenderPass.hpp"
#include "zensim/vulkan/Vulkan.hpp"
#include "zensim/zpc_tpls/magic_enum/magic_enum.hpp"

namespace zs {

  /**
    ref: 
    CPU: ANI  ... QS   ... QP         ANI  ... QS   ... QP
       S:S1     W:S1     W:S2       S:S3     W:S3     W:S4
       S:F1     S:S2                S:F2     S:S4
    GPU:          <------ R ------>            <------ R ------>
    PE:                           <-- P -->                    <-- P -->
  */
  ///
  ///
  /// swapchain builder
  ///
  ///
  u32 Swapchain::newFrame() noexcept {
    frameIndex = (frameIndex + 1) % num_buffered_frames;
    if (currentFence()) {
      if (vk::Result res = ctx.device.waitForFences(
              1, &currentFence(), VK_TRUE, detail::deduce_numeric_max<u64>(), ctx.dispatcher);
          res != vk::Result::eSuccess)
        throw std::runtime_error(
            fmt::format("[newFrame]: Failed to wait for fence at frame [{}] with result [{}]\n",
                        frameIndex, magic_enum::enum_name(res)));
      recycleFence(currentFence());
      recycleSemaphore(currentImageAcquiredSemaphore());
      // for (auto& garbage : currentSwapchainGarbage()) cleanupSwapchainObjects(garbage);
      currentSwapchainGarbage().clear();
      // submission have finished execution, but no guarantee rendercomplete semaphore is not in use
      // present semaphore is set to VK_NULL_HANDLE after queue present op
      assert(currentRenderCompleteSemaphore() == VK_NULL_HANDLE);
    }
    currentFence() = getFence();
    currentImageAcquiredSemaphore() = getSemaphore();
    currentRenderCompleteSemaphore() = getSemaphore();
    return frameIndex;
  }
  vk::Result Swapchain::acquireNextImage(u32& imageId) {
#if 0
    auto res = ctx.device.acquireNextImageKHR(
        swapchain, detail::deduce_numeric_max<u64>(),
        currentImageAcquiredSemaphore(),  // must be a not signaled semaphore
        VK_NULL_HANDLE, ctx.dispatcher);
    imageId = res.value;
    return res.result;
#endif
    vk::Fence acquireFence = getFence();
    auto result = ctx.dispatcher.vkAcquireNextImageKHR(
        (vk::Device)ctx.device, (vk::SwapchainKHR)swapchain, detail::deduce_numeric_max<u64>(),
        currentImageAcquiredSemaphore(), acquireFence, &imageId);

    if (result != VK_SUCCESS) {
      recycleFence(acquireFence);
    } else {
      for (size_t i = 0; i < presentHistory.size(); ++i) {
        auto& presentInfo = presentHistory[presentHistory.size() - 1 - i];
        // the remaining marked for destruction upon swapchain recreation
        if (presentInfo.imageIndex == s_invalid_image_index)
          break;
        if (presentInfo.imageIndex == imageId) {
          assert(presentInfo.cleanupFence == VK_NULL_HANDLE
                 && fmt::format("the present record with the matching image index should not "
                                "already have an associated fence")
                        .c_str());
          // associate fence with the present operation matching the image index
          presentInfo.cleanupFence = acquireFence;
          return vk::Result{result};
        }
      }
      // if none associated, add fence with present operation
      presentHistory.push_back(PresentOperationInfo{acquireFence, VK_NULL_HANDLE, {}, imageId});
    }
    return vk::Result{result};
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
    vk::Result res = static_cast<vk::Result>(ctx.dispatcher.vkQueuePresentKHR(
        static_cast<VkQueue>(queue), reinterpret_cast<const VkPresentInfoKHR*>(&presentInfo)));
    /// add present to history
    presentHistory.push_back(PresentOperationInfo{VK_NULL_HANDLE, currentRenderCompleteSemaphore(),
                                                  std::move(oldSwapchains), imageId});
    currentRenderCompleteSemaphore() = VK_NULL_HANDLE;  // set to null after present
    /// cleanup present history
    cleanupPresentHistory();
    return res;
  }
  void Swapchain::cleanupPresentHistory() {
    while (!presentHistory.empty()) {
      PresentOperationInfo& presentInfo = presentHistory.front();
      // If there is no fence associated with the history, it can't be cleaned up yet.
      if (presentInfo.cleanupFence == VK_NULL_HANDLE) {
        // Can't have an old present operation without a fence that doesn't have an
        // image index used to later associate a fence with it.
        assert(presentInfo.imageIndex != s_invalid_image_index);
        break;
      }
      // check if fence is signaled
      auto result = ctx.device.getFenceStatus(presentInfo.cleanupFence, ctx.dispatcher);
      if (result == vk::Result::eNotReady) break;

      cleanupPresentInfo(presentInfo);
      presentHistory.pop_front();
    }
    if (presentHistory.size() > images.size() * 2
        && presentHistory.front().cleanupFence == VK_NULL_HANDLE) {
      PresentOperationInfo presentInfo = std::move(presentHistory.front());
      presentHistory.pop_front();
      assert(presentInfo.imageIndex != s_invalid_image_index);
      presentHistory.front().oldSwapchains
          = std::move(presentInfo.oldSwapchains);        // transfer old swapchains
      presentHistory.push_back(std::move(presentInfo)); // moved to tail
    }
  }
  void Swapchain::cleanupPresentInfo(PresentOperationInfo& presentInfo) {
    if (presentInfo.cleanupFence) recycleFence(presentInfo.cleanupFence);
    if (presentInfo.presentSemaphore) recycleSemaphore(presentInfo.presentSemaphore);
    for (auto& oldSwapchain : presentInfo.oldSwapchains) cleanupOldSwapchain(oldSwapchain);
  }
  void Swapchain::cleanupOldSwapchain(SwapchainCleanupData& oldSwapchain) {
    if (oldSwapchain.swapchain)
      ctx.device.destroySwapchainKHR(oldSwapchain.swapchain, nullptr, ctx.dispatcher);
    for (auto semaphore : oldSwapchain.semaphores)
      ctx.device.destroySemaphore(semaphore, nullptr, ctx.dispatcher);
    oldSwapchain = {};
  }

  /**
   * @brief When a swapchain is retired, the resources associated with its images are scheduled to
   * be cleaned up as soon as the last submission using those images is complete. This function is
   * called at such a moment.
   * The swapchain itself is not destroyed until known safe.
   */
  void Swapchain::cleanupSwapchainObjects(SwapchainObjects& garbage) { garbage = {}; }

  /**
   * @brief The previous swapchain which needs to be scheduled for destruction when
   * appropriate.  This will be done when the first image of the current swapchain is
   * presented.  If there were older swapchains pending destruction when the swapchain is
   * recreated, they will accumulate and be destroyed with the previous swapchain.
   *
   * Note that if the user resizes the window such that the swapchain is recreated every
   * frame, this array can go grow indefinitely.
   */
  void Swapchain::scheduleOldSwapchainForDestruction(vk::SwapchainKHR oldSwapchain) {
    if (!presentHistory.empty() && presentHistory.back().imageIndex == s_invalid_image_index) {
      ctx.device.destroySwapchainKHR(oldSwapchain, nullptr, ctx.dispatcher);
      return;
    }
    SwapchainCleanupData cleanup;
    cleanup.swapchain = oldSwapchain;

    // Place any present operation that's not associated with a fence into oldSwapchains.
    // That gets scheduled for destruction when the semaphore of the first image of the next
    // swapchain can be recycled.
    std::vector<PresentOperationInfo> historyToKeep;
    while (!presentHistory.empty()) {
      PresentOperationInfo& presentInfo = presentHistory.back();

      // If this is about an older swapchain, let it be.
      if (presentInfo.imageIndex == s_invalid_image_index) {
        assert(presentInfo.cleanupFence != VK_NULL_HANDLE);
        break;
      }

      // Reset the index, so it's not processed in the future.
      presentInfo.imageIndex = s_invalid_image_index;
      if (presentInfo.cleanupFence != VK_NULL_HANDLE) {
        // If there is already a fence associated, let it be cleaned up once the fence is signaled.
        historyToKeep.push_back(std::move(presentInfo));
      } else {
        assert(presentInfo.presentSemaphore != VK_NULL_HANDLE);
        // Otherwise accumulate it in cleanup data.
        cleanup.semaphores.push_back(presentInfo.presentSemaphore);
        // Accumulate any previous swapchains that are pending destruction too.
        for (SwapchainCleanupData& swapchainCleanup : presentInfo.oldSwapchains)
          oldSwapchains.emplace_back(swapchainCleanup);
        presentInfo.oldSwapchains.clear();
      }

      presentHistory.pop_back();
    }
    std::move(historyToKeep.begin(), historyToKeep.end(), std::back_inserter(presentHistory));

    if (cleanup.swapchain != VK_NULL_HANDLE || !cleanup.semaphores.empty())
      oldSwapchains.emplace_back(std::move(cleanup));
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

  void Swapchain::initFramebuffersFor(vk::RenderPass renderPass) {
    const u32 cnt = imageCount();
    swapchainObjects.frameBuffers.clear();
    swapchainObjects.frameBuffers.reserve(cnt);
    const bool enableMS = multiSampleEnabled();
    const bool enableDepth = depthEnabled();
    std::vector<vk::ImageView> imgvs;
    const auto dim = (1 + (enableMS ? 1 : 0) + (enableDepth ? 1 : 0));
    for (u32 i = 0; i != cnt; ++i) {
      imgvs.clear();
      imgvs.reserve(dim);
      if (enableMS) imgvs.push_back((vk::ImageView)swapchainObjects.msColorBuffers[i]);
      imgvs.push_back((vk::ImageView)swapchainObjects.imageViews[i]);
      if (enableDepth) imgvs.push_back((vk::ImageView)swapchainObjects.depthBuffers[i]);

      swapchainObjects.frameBuffers.emplace_back(ctx.createFramebuffer(imgvs, ci.imageExtent, renderPass));
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
  void SwapchainBuilder::resize(Swapchain& obj, u32 width, u32 height, vk::RenderPass renderPass) {
    /// @note credits
    /// https://www.reddit.com/r/vulkan/comments/cc3edr/swapchain_recreation_repeatedly_returns_vk_error/
    surfCapabilities = ctx.physicalDevice.getSurfaceCapabilitiesKHR(obj.ci.surface, ctx.dispatcher);
    width = std::clamp(width, surfCapabilities.minImageExtent.width,
                       surfCapabilities.maxImageExtent.width);
    height = std::clamp(height, surfCapabilities.minImageExtent.height,
                        surfCapabilities.maxImageExtent.height);
    obj.ci.imageExtent = vk::Extent2D{width, height};

    build(obj, renderPass);
  }
  SwapchainBuilder& SwapchainBuilder::presentMode(vk::PresentModeKHR mode) {
    if (mode == ci.presentMode) return *this;
    for (auto& m : surfPresentModes)
      if (m == mode) {
        ci.presentMode = mode;
        return *this;
      }
    ZS_WARN(fmt::format("Present mode [{}] is not supported in this context. Ignored.\n",
                        magic_enum::enum_name(mode)));
    return *this;
  }
  void SwapchainBuilder::build(Swapchain& obj, vk::RenderPass renderPass) {
    // kept the previously built swapchain for this
    const bool rebuild = obj.swapchain != VK_NULL_HANDLE;
    if (rebuild) ci = obj.ci;
    ci.oldSwapchain = obj.swapchain;
    obj.ci = ci;
    obj.swapchain = ctx.device.createSwapchainKHR(ci, nullptr, ctx.dispatcher);

    if (rebuild) {
      obj.scheduleOldSwapchainForDestruction(ci.oldSwapchain); 
    } else {
      obj.depthFormat = swapchainDepthFormat;
      obj.sampleBits = sampleBits;
    }

    // schedule destruction of the old swapchain resources once this frame's submission is finished.
    auto& swapchainObjects = obj.swapchainObjects;
    // to avoid initial access violation of frame index
    if (obj.getCurrentFrame() >= 0)
      obj.currentSwapchainGarbage().push_back(std::move(swapchainObjects));

    obj.images = ctx.device.getSwapchainImagesKHR(obj.swapchain, ctx.dispatcher);
    const auto numSwapchainImages = obj.images.size();

    // construct current swapchain's objects (except framebuffers)
    // swapchainObjects is now reset (moved from)
    swapchainObjects.imageViews.reserve(numSwapchainImages);
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
      swapchainObjects.imageViews.emplace_back(
          ctx, ctx.device.createImageView(ivCI, nullptr, ctx.dispatcher));
    }
    if (obj.sampleBits != vk::SampleCountFlagBits::e1) {
      swapchainObjects.msColorBuffers.reserve(numSwapchainImages);
      for (int i = 0; i != numSwapchainImages; ++i) {
        swapchainObjects.msColorBuffers.emplace_back(ctx.create2DImage(
            ci.imageExtent, ci.imageFormat,
            vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal, /*mipmaps*/ false, /*createView*/ true,
            /*enable transfer*/ false, obj.sampleBits));
      }
    }
    if (buildDepthBuffer) {
      swapchainObjects.depthBuffers.reserve(numSwapchainImages);
      for (int i = 0; i != numSwapchainImages; ++i) {
        swapchainObjects.depthBuffers.emplace_back(ctx.create2DImage(
            ci.imageExtent, obj.depthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal, /*mipmaps*/ false, /*createView*/ true,
            /*enable transfer*/ false, obj.sampleBits));
      }
    }

    if (renderPass != VK_NULL_HANDLE) obj.initFramebuffersFor(renderPass);
  }

}  // namespace zs