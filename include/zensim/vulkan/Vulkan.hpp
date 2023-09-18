#pragma once

#include <cstdint>
//
#include "zensim/vulkan/VkContext.hpp"
//
#include "zensim/Reflection.h"
#include "zensim/Singleton.h"
#include "zensim/profile/CppTimers.hpp"
#include "zensim/types/SourceLocation.hpp"

namespace zs {

  struct Vulkan : Singleton<Vulkan> {
  public:
    Vulkan();
    ~Vulkan();

    static auto &driver() noexcept { return instance(); }
    static size_t num_devices() noexcept { return instance()._contexts.size(); }
    static vk::Instance vk_inst() noexcept { return instance()._instance; }
    static const vk::DispatchLoaderDynamic &vk_inst_dispatcher() noexcept {
      return instance()._dispatcher;
    }
    static auto &context(int devid) { return driver()._contexts[devid]; }
    static auto &context() { return instance()._contexts[instance()._defaultContext]; }

  private:
    vk::Instance _instance;
    vk::DispatchLoaderDynamic _dispatcher;  // store vulkan-instance calls
    vk::DebugUtilsMessengerEXT _messenger;
    std::vector<VulkanContext> _contexts;  ///< generally one per device
    int _defaultContext = 0;
  };

  u32 check_current_working_contexts();

  struct Swapchain {
    /// triple buffering
    static constexpr u32 num_buffered_frames = 3;

    Swapchain() = delete;
    Swapchain(VulkanContext &ctx) : ctx{ctx}, swapchain{} {}
    Swapchain(const Swapchain &) = delete;
    Swapchain(Swapchain &&o) noexcept
        : ctx{o.ctx},
          swapchain{o.swapchain},
          images{std::move(o.images)},
          imageViews{std::move(o.imageViews)},
          readSemaphores{std::move(o.readSemaphores)},
          writeSemaphores{std::move(o.writeSemaphores)},
          readFences{std::move(o.readFences)},
          writeFences{std::move(o.writeFences)},
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
    void nextFrame() { frameIndex = (frameIndex + 1) % num_buffered_frames; }
    RenderPass getRenderPass();
    void initFramebuffersFor(vk::RenderPass renderPass);

    // update width, height
    vk::SwapchainKHR operator*() const { return swapchain; }
    operator vk::SwapchainKHR() const { return swapchain; }

  protected:
    void resetAux() {
      for (auto &v : imageViews) ctx.device.destroyImageView(v, nullptr, ctx.dispatcher);
      imageViews.clear();
      depthBuffers.clear();
      frameBuffers.clear();
      resetSyncPrimitives();
    }
    void resetSyncPrimitives() {
      for (auto &s : readSemaphores) ctx.device.destroySemaphore(s, nullptr, ctx.dispatcher);
      readSemaphores.clear();
      for (auto &s : writeSemaphores) ctx.device.destroySemaphore(s, nullptr, ctx.dispatcher);
      writeSemaphores.clear();
      for (auto &f : readFences) ctx.device.destroyFence(f, nullptr, ctx.dispatcher);
      readFences.clear();
      for (auto &f : writeFences) ctx.device.destroyFence(f, nullptr, ctx.dispatcher);
      writeFences.clear();
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
    std::vector<vk::Semaphore> readSemaphores;
    std::vector<vk::Semaphore> writeSemaphores;
    std::vector<vk::Fence> readFences;
    std::vector<vk::Fence> writeFences;
    int frameIndex;
  };

  // ref: LegitEngine (https://github.com/Raikiri/LegitEngine), nvpro_core
  struct SwapchainBuilder {
    SwapchainBuilder(VulkanContext &ctx, vk::SurfaceKHR targetSurface);
    SwapchainBuilder(const SwapchainBuilder &) = delete;
    SwapchainBuilder(SwapchainBuilder &&) noexcept = default;
    ~SwapchainBuilder() {
      zs::Vulkan::vk_inst().destroySurfaceKHR(surface, nullptr, zs::Vulkan::vk_inst_dispatcher());
    }

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

  SwapchainBuilder &VulkanContext::swapchain(vk::SurfaceKHR surface, bool reset) {
    if (!swapchainBuilder || reset || swapchainBuilder->getSurface() != surface)
      swapchainBuilder.reset(new SwapchainBuilder(*this, surface));
    return *swapchainBuilder;
  }

}  // namespace zs
