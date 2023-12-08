#pragma once
#include "zensim/vulkan/VkContext.hpp"

namespace zs {

  struct Swapchain {
    /// triple buffering

    Swapchain() = delete;
    Swapchain(VulkanContext &ctx) : ctx{ctx}, swapchain{VK_NULL_HANDLE} {}
    Swapchain(const Swapchain &) = delete;
    Swapchain(Swapchain &&o) noexcept
        : ctx{o.ctx},
          swapchain{o.swapchain},
#if 0
          extent{o.extent},
          colorFormat{o.colorFormat},
          depthFormat{o.depthFormat},
          imageColorSpace{o.imageColorSpace},
          presentMode{o.presentMode},
#else
          depthFormat{o.depthFormat},
          ci{o.ci},
#endif
          //
          images{std::move(o.images)},
          imageViews{std::move(o.imageViews)},
          depthBuffers{std::move(o.depthBuffers)},
          frameBuffers{std::move(o.frameBuffers)},
          // sync prims
          imageAcquiredSemaphores{std::move(o.imageAcquiredSemaphores)},
          renderCompleteSemaphores{std::move(o.renderCompleteSemaphores)},
          fences{std::move(o.fences)},
          imageFences{std::move(o.imageFences)},
          frameIndex(o.frameIndex) {
      o.swapchain = VK_NULL_HANDLE;
      o.frameIndex = 0;
    }
    ~Swapchain() {
      resetAux();
      ctx.device.destroySwapchainKHR(swapchain, nullptr, ctx.dispatcher);
    }

    std::vector<vk::Image> getImages() const { return images; }
    u32 imageCount() const { return images.size(); }
    vk::ImageView getColorView(u32 i) const { return imageViews[i]; }
    const Image &getDepthView(u32 i) const { return depthBuffers[i]; }

    vk::Extent2D getExtent() const noexcept { return ci.imageExtent; }
    vk::Format getColorFormat() const noexcept { return ci.imageFormat; }
    vk::Format getDepthFormat() const noexcept { return depthFormat; }
    vk::PresentModeKHR getPresentMode() const noexcept { return ci.presentMode; }
    vk::ColorSpaceKHR getImageColorSpace() const noexcept { return ci.imageColorSpace; }

    vk::Result acquireNextImage(u32 &imageId);
    u32 getCurrentFrame() const noexcept { return frameIndex; }
    u32 nextFrame() noexcept { return frameIndex = (frameIndex + 1) % num_buffered_frames; }
    void initFramebuffersFor(vk::RenderPass renderPass);
    void resetFramebuffers(const std::vector<vk::Framebuffer> &fbs);

    RenderPass getRenderPass();
    Framebuffer &frameBuffer(u32 imageIndex) { return frameBuffers[imageIndex]; }

    vk::Fence &imageFence(u32 id) noexcept { return imageFences[id]; }
    vk::Fence &currentFence() noexcept { return fences[frameIndex]; }
    vk::Semaphore &currentImageAcquiredSemaphore() noexcept {
      return imageAcquiredSemaphores[frameIndex];
    }
    vk::Semaphore &currentRenderCompleteSemaphore() noexcept {
      return renderCompleteSemaphores[frameIndex];
    }

    // update width, height
    vk::SwapchainKHR operator*() const { return swapchain; }
    operator vk::SwapchainKHR() const { return swapchain; }

  protected:
    void resetAux() {
      images.clear();

      frameBuffers.clear();
      for (auto &v : imageViews) ctx.device.destroyImageView(v, nullptr, ctx.dispatcher);
      imageViews.clear();
      depthBuffers.clear();
      resetSyncPrimitives();
    }
    void resetSyncPrimitives() {
      imageFences.clear();

      for (auto &s : imageAcquiredSemaphores)
        ctx.device.destroySemaphore(s, nullptr, ctx.dispatcher);
      imageAcquiredSemaphores.clear();
      for (auto &s : renderCompleteSemaphores)
        ctx.device.destroySemaphore(s, nullptr, ctx.dispatcher);
      renderCompleteSemaphores.clear();
      for (auto &f : fences) ctx.device.destroyFence(f, nullptr, ctx.dispatcher);
      fences.clear();
    }
    friend struct SwapchainBuilder;

    VulkanContext &ctx;
    vk::SwapchainKHR swapchain;
#if 0
    vk::Extent2D extent;
    vk::Format colorFormat, depthFormat;
    vk::ColorSpaceKHR imageColorSpace;
    vk::PresentModeKHR presentMode;
#else
    vk::Format depthFormat;
    vk::SwapchainCreateInfoKHR ci;
#endif
    ///
    std::vector<vk::Image> images;
    std::vector<vk::ImageView> imageViews;  // corresponds to [images] from swapchain
    std::vector<Image> depthBuffers;        // optional
    std::vector<Framebuffer> frameBuffers;  // initialized later
    ///
    // littleVulkanEngine-alike setup
    std::vector<vk::Semaphore> imageAcquiredSemaphores;   // ready to read
    std::vector<vk::Semaphore> renderCompleteSemaphores;  // ready to write
    std::vector<vk::Fence> fences;                        // ready to submit
    std::vector<vk::Fence> imageFences;                   // directed to the above 'fences' objects
    int frameIndex;
  };

  // ref: LegitEngine (https://github.com/Raikiri/LegitEngine), nvpro_core
  struct SwapchainBuilder {
    SwapchainBuilder(VulkanContext &ctx, vk::SurfaceKHR targetSurface);
    SwapchainBuilder(const SwapchainBuilder &) = delete;
    SwapchainBuilder(SwapchainBuilder &&) noexcept = default;
    ~SwapchainBuilder() = default;

    vk::SurfaceKHR getSurface() const { return surface; }

    SwapchainBuilder &imageCount(u32 count) {
      ci.minImageCount = std::clamp(count, (u32)surfCapabilities.minImageCount,
                                    (u32)surfCapabilities.maxImageCount);
      return *this;
    }
    SwapchainBuilder &extent(u32 width, u32 height) {
      ci.imageExtent = vk::Extent2D{width, height};
      return *this;
    }
    SwapchainBuilder &presentMode(vk::PresentModeKHR mode = vk::PresentModeKHR::eMailbox);
    SwapchainBuilder &usage(vk::ImageUsageFlags flags) {
      ci.imageUsage = flags;
      return *this;
    }
    SwapchainBuilder &enableDepth() {
      buildDepthBuffer = true;
      return *this;
    }

    Swapchain build() {
      Swapchain obj(ctx);
      build(obj);
      return obj;
    }
    void resize(Swapchain &obj, u32 width, u32 height);

  private:
    void build(Swapchain &obj);

    VulkanContext &ctx;
    vk::SurfaceKHR surface;
    std::vector<vk::SurfaceFormatKHR> surfFormats;
    std::vector<vk::PresentModeKHR> surfPresentModes;
    vk::SurfaceCapabilitiesKHR surfCapabilities;
    vk::Format swapchainDepthFormat;

    vk::SwapchainCreateInfoKHR ci;
    bool buildDepthBuffer{false};
  };

}  // namespace zs