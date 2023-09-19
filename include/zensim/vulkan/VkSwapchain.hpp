#pragma once
#include "zensim/vulkan/VkContext.hpp"

namespace zs {

  struct Swapchain {
    /// triple buffering
    static constexpr u32 num_buffered_frames = 3;

    Swapchain() = delete;
    Swapchain(VulkanContext &ctx) : ctx{ctx}, swapchain{} {}
    Swapchain(const Swapchain &) = delete;
    Swapchain(Swapchain &&o) noexcept
        : ctx{o.ctx},
          swapchain{o.swapchain},
          extent{o.extent},
          colorFormat{o.colorFormat},
          depthFormat{o.depthFormat},
          imageColorSpace{o.imageColorSpace},
          presentMode{o.presentMode},
          //
          images{std::move(o.images)},
          imageViews{std::move(o.imageViews)},
          depthBuffers{std::move(o.depthBuffers)},
          frameBuffers{std::move(o.frameBuffers)},
          // sync prims
          readSemaphores{std::move(o.readSemaphores)},
          writeSemaphores{std::move(o.writeSemaphores)},
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
    vk::Extent2D getExtent() const noexcept { return extent; }
    vk::Format getColorFormat() const noexcept { return colorFormat; }
    vk::Format getDepthFormat() const noexcept { return depthFormat; }
    vk::PresentModeKHR getPresentMode() const noexcept { return presentMode; }
    vk::ColorSpaceKHR getImageColorSpace() const noexcept { return imageColorSpace; }

    u32 acquireNextImage();
    u32 getCurrentFrame() const noexcept { return frameIndex; }
    u32 nextFrame() noexcept { return frameIndex = (frameIndex + 1) % num_buffered_frames; }
    void initFramebuffersFor(vk::RenderPass renderPass);

    RenderPass getRenderPass();
    Framebuffer &frameBuffer(u32 imageIndex) { return frameBuffers[imageIndex]; }

    vk::Fence &imageFence(u32 id) noexcept { return imageFences[id]; }
    vk::Fence &currentFence() noexcept { return fences[frameIndex]; }
    vk::Semaphore &currentReadSemaphore() noexcept { return readSemaphores[frameIndex]; }
    vk::Semaphore &currentWriteSemaphore() noexcept { return writeSemaphores[frameIndex]; }

    // update width, height
    vk::SwapchainKHR operator*() const { return swapchain; }
    operator vk::SwapchainKHR() const { return swapchain; }

  protected:
    void resetAux() {
      images.clear();

      for (auto &v : imageViews) ctx.device.destroyImageView(v, nullptr, ctx.dispatcher);
      imageViews.clear();
      depthBuffers.clear();
      frameBuffers.clear();
      resetSyncPrimitives();
    }
    void resetSyncPrimitives() {
      imageFences.clear();

      for (auto &s : readSemaphores) ctx.device.destroySemaphore(s, nullptr, ctx.dispatcher);
      readSemaphores.clear();
      for (auto &s : writeSemaphores) ctx.device.destroySemaphore(s, nullptr, ctx.dispatcher);
      writeSemaphores.clear();
      for (auto &f : fences) ctx.device.destroyFence(f, nullptr, ctx.dispatcher);
      fences.clear();
    }
    friend struct SwapchainBuilder;

    VulkanContext &ctx;
    vk::SwapchainKHR swapchain;
    vk::Extent2D extent;
    vk::Format colorFormat, depthFormat;
    vk::ColorSpaceKHR imageColorSpace;
    vk::PresentModeKHR presentMode;
    ///
    std::vector<vk::Image> images;
    std::vector<vk::ImageView> imageViews;
    std::vector<Image> depthBuffers;        // optional
    std::vector<Framebuffer> frameBuffers;  // initialized later
    ///
    // littleVulkanEngine-alike setup
    std::vector<vk::Semaphore> readSemaphores;   // ready to read
    std::vector<vk::Semaphore> writeSemaphores;  // ready to write
    std::vector<vk::Fence> fences;               // ready to submit
    std::vector<vk::Fence> imageFences;          // directed to the above 'fences' objects
    int frameIndex;
  };

  // ref: LegitEngine (https://github.com/Raikiri/LegitEngine), nvpro_core
  struct SwapchainBuilder {
    SwapchainBuilder(VulkanContext &ctx, vk::SurfaceKHR targetSurface);
    SwapchainBuilder(const SwapchainBuilder &) = delete;
    SwapchainBuilder(SwapchainBuilder &&) noexcept = default;
    ~SwapchainBuilder();

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

    void build(Swapchain &obj);
    Swapchain build() {
      Swapchain obj(ctx);
      build(obj);
      return obj;
    }
    void resize(Swapchain &obj, u32 width, u32 height) {
      width = std::clamp(width, surfCapabilities.minImageExtent.width,
                         surfCapabilities.maxImageExtent.width);
      height = std::clamp(height, surfCapabilities.minImageExtent.height,
                          surfCapabilities.maxImageExtent.height);
      ci.imageExtent = vk::Extent2D{width, height};

      build(obj);
    }

  private:
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