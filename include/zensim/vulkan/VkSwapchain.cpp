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
    auto res = ctx.device.acquireNextImageKHR(
        swapchain, detail::deduce_numeric_max<u64>(),
        currentImageAcquiredSemaphore(),  // must be a not signaled semaphore
        VK_NULL_HANDLE, ctx.dispatcher);
    imageId = res.value;
    return res.result;
#if 0
    if (res.result != vk::Result::eSuccess)
      throw std::runtime_error(fmt::format(
          "[acquireNextImage]: Failed to acquire next image at frame [{}] with result [{}]\n",
          frameIndex, res.result));
    return res.value;
#endif
  }
  RenderPass Swapchain::getRenderPass() {
    const bool includeDepthBuffer = depthBuffers.size() == imageCount() && depthBuffers.size() != 0;

    auto rpBuilder = ctx.renderpass()
                         .addAttachment(ci.imageFormat, vk::ImageLayout::eUndefined,
                                        vk::ImageLayout::ePresentSrcKHR, true)
                         .setNumPasses(1);
    if (includeDepthBuffer)
      rpBuilder.addAttachment(depthFormat, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eDepthStencilAttachmentOptimal, true);
    return rpBuilder.build();
  }
  void Swapchain::initFramebuffersFor(vk::RenderPass renderPass) {
    frameBuffers.clear();
    auto cnt = imageCount();
    if (depthBuffers.size() == cnt) {
      // color + depth
      for (int i = 0; i != cnt; ++i) {
        frameBuffers.emplace_back(
            ctx.createFramebuffer({(vk::ImageView)imageViews[i], (vk::ImageView)depthBuffers[i]},
                                  ci.imageExtent, renderPass));
      }
    } else {
      // color only
      for (int i = 0; i != imageCount(); ++i) {
        frameBuffers.emplace_back(
            ctx.createFramebuffer({(vk::ImageView)imageViews[i]}, ci.imageExtent, renderPass));
      }
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
    if (obj.swapchain != VK_NULL_HANDLE) ci = obj.ci;
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
    obj.depthFormat = swapchainDepthFormat;
    obj.ci = ci;
    // if (obj.swapchain == VK_NULL_HANDLE)
    //   obj.ci = ci;
    // else
    //   obj.ci.oldSwapchain = ci.oldSwapchain;
#endif

    obj.resetAux();
    obj.images = ctx.device.getSwapchainImagesKHR(obj.swapchain, ctx.dispatcher);

    /// reset previous resources (if any)
    ctx.device.destroySwapchainKHR(ci.oldSwapchain, nullptr, ctx.dispatcher);

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
    obj.imageFences.resize(obj.images.size(), VK_NULL_HANDLE);
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