#pragma once
#include <deque>
#include "zensim/vulkan/VkContext.hpp"
#include "VkImage.hpp"

namespace zs {

  struct ZPC_CORE_API Swapchain {
    /// triple buffering

    Swapchain() = delete;
    Swapchain(VulkanContext &ctx)
        : ctx{ctx},
          swapchain{VK_NULL_HANDLE} {}
    Swapchain(const Swapchain &) = delete;
    Swapchain(Swapchain &&o) noexcept
        : ctx{o.ctx},
          swapchain{std::exchange(o.swapchain, VK_NULL_HANDLE)},
          depthFormat{o.depthFormat},
          sampleBits{o.sampleBits},
          ci{o.ci},
          //
          images{std::move(o.images)},
          swapchainObjects{std::move(o.swapchainObjects)},
          // sync prims
          imageAcquiredSemaphores{std::move(o.imageAcquiredSemaphores)},
          renderCompleteSemaphores{std::move(o.renderCompleteSemaphores)},
          submitFences{std::move(o.submitFences)},
          //
          fencePool{std::move(o.fencePool)},
          semaphorePool{std::move(o.semaphorePool)},
          presentHistory{std::move(o.presentHistory)},
          swapchainGarbages{std::move(o.swapchainGarbages)},
          oldSwapchains(std::move(o.oldSwapchains)),
          //
          frameIndex(std::exchange(o.frameIndex, -1)) {
    }
    ~Swapchain() {
      ctx.device.waitIdle(ctx.dispatcher);
      // cleanup buffers
      for (const auto &f : submitFences) ctx.device.destroyFence(f, nullptr, ctx.dispatcher);
      for (const auto &s : imageAcquiredSemaphores)
        ctx.device.destroySemaphore(s, nullptr, ctx.dispatcher);
      for (auto &garbages : swapchainGarbages) garbages.clear();
      for (const auto &s : renderCompleteSemaphores) assert(s == VK_NULL_HANDLE);
      // cleanup present history
      for (PresentOperationInfo &presentInfo : presentHistory) {
        if (presentInfo.cleanupFence != VK_NULL_HANDLE)
          ctx.device.waitForFences(1, &presentInfo.cleanupFence, true, UINT64_MAX, ctx.dispatcher);
        cleanupPresentInfo(presentInfo);
      }
      // cleanup old swapchains
      for (SwapchainCleanupData &oldSwapchain : oldSwapchains) cleanupOldSwapchain(oldSwapchain);
      // cleanup swapchain objects
      cleanupSwapchainObjects(swapchainObjects);
      // cleanup sync primitive pools
      for (const auto &f : fencePool) ctx.device.destroyFence(f, nullptr, ctx.dispatcher);
      for (const auto &s : semaphorePool) ctx.device.destroySemaphore(s, nullptr, ctx.dispatcher);
      ctx.device.destroySwapchainKHR(swapchain, nullptr, ctx.dispatcher);
      swapchain = VK_NULL_HANDLE;
      frameIndex = -1;
    }

    std::vector<vk::Image> getImages() const { return images; }
    u32 imageCount() const { return images.size(); }
    vk::ImageView getColorView(u32 i) const { return swapchainObjects.imageViews[i]; }
    const Image &getMsColorView(u32 i) const { return swapchainObjects.msColorBuffers[i]; }
    const Image &getDepthView(u32 i) const { return swapchainObjects.depthBuffers[i]; }

    bool depthEnabled() const noexcept { return swapchainObjects.depthBuffers.size() > 0; }
    bool multiSampleEnabled() const noexcept { return sampleBits != vk::SampleCountFlagBits::e1; }
    vk::Extent2D getExtent() const noexcept { return ci.imageExtent; }
    vk::Format getColorFormat() const noexcept { return ci.imageFormat; }
    vk::Format getDepthFormat() const noexcept { return depthFormat; }
    vk::SampleCountFlagBits getSampleBits() const noexcept { return sampleBits; }
    vk::PresentModeKHR getPresentMode() const noexcept { return ci.presentMode; }
    vk::ColorSpaceKHR getImageColorSpace() const noexcept { return ci.imageColorSpace; }
    std::vector<vk::ClearValue> getClearValues() const noexcept {
      vk::ClearValue val;
      std::vector<vk::ClearValue> vals;
      if (multiSampleEnabled()) {
        val.color = vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}};
        vals.push_back(val);
      }
      val.color = vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}};
      vals.push_back(val);
      if (depthEnabled()) {
        vk::ClearValue val;
        val.depthStencil = vk::ClearDepthStencilValue{1.f, 0};
        vals.push_back(val);
      }
      return vals;
    }

    u32 newFrame() noexcept;
    vk::Result acquireNextImage(u32 &imageId);
    vk::Result present(vk::Queue queue, u32 imageId);
    auto getCurrentFrame() const noexcept { return frameIndex; }

    vk::Fence &currentFence() noexcept { return submitFences[frameIndex]; }
    vk::Semaphore &currentImageAcquiredSemaphore() noexcept {
      return imageAcquiredSemaphores[frameIndex];
    }
    vk::Semaphore &currentRenderCompleteSemaphore() noexcept {
      return renderCompleteSemaphores[frameIndex];
    }

    /// @brief not expected to be called upon swpachain recreation
    /// only valid use is when only the renderpass has changed
    void initFramebuffersFor(vk::RenderPass renderPass);

    RenderPass getRenderPass();
    Framebuffer &frameBuffer(u32 imageIndex) { return swapchainObjects.frameBuffers[imageIndex]; }

    // update width, height
    vk::SwapchainKHR operator*() const { return swapchain; }
    operator vk::SwapchainKHR() const { return swapchain; }

  protected:
    friend struct SwapchainBuilder;

    VulkanContext &ctx;
    vk::SwapchainKHR swapchain;
    vk::Format depthFormat;
    vk::SampleCountFlagBits sampleBits;
    vk::SwapchainCreateInfoKHR ci;

    /// per-image data
    struct SwapchainObjects {
      std::vector<ImageView> imageViews;      // corresponds to [images] from swapchain
      std::vector<Image> msColorBuffers;      // optional
      std::vector<Image> depthBuffers;        // optional
      std::vector<Framebuffer> frameBuffers;  // initialized later
    };
    struct SwapchainCleanupData {
      vk::SwapchainKHR swapchain{VK_NULL_HANDLE};
      std::vector<vk::Semaphore> semaphores;
    };

    std::vector<vk::Image> images;
    SwapchainObjects swapchainObjects;  // swapchain objects

    /// per-frame data
    std::array<vk::Semaphore, num_buffered_frames> imageAcquiredSemaphores;   // ready to read
    std::array<vk::Semaphore, num_buffered_frames> renderCompleteSemaphores;  // ready to write
    std::array<vk::Fence, num_buffered_frames> submitFences;                  // ready to submit

    /// fence pool used during acquireNextImage and present operations.
    std::vector<vk::Fence> fencePool;
    void recycleFence(vk::Fence fence) {
      fencePool.push_back(fence);
      ctx.device.resetFences(1, &fence, ctx.dispatcher);
    }
    vk::Fence getFence() {
      if (!fencePool.empty()) {
        auto ret = fencePool.back();
        fencePool.pop_back();
        return ret;
      }
      return ctx.device.createFence(vk::FenceCreateInfo{}, nullptr, ctx.dispatcher);
    }

    /// semaphore pool
    std::vector<vk::Semaphore> semaphorePool;
    void recycleSemaphore(vk::Semaphore semaphore) { semaphorePool.push_back(semaphore); }
    vk::Semaphore getSemaphore() {
      if (!semaphorePool.empty()) {
        auto ret = semaphorePool.back();
        semaphorePool.pop_back();
        return ret;
      }
      return ctx.device.createSemaphore(vk::SemaphoreCreateInfo{}, nullptr, ctx.dispatcher);
    }

    /// history info
    static constexpr u32 s_invalid_image_index = ~static_cast<u32>(0);
    struct PresentOperationInfo {
      vk::Fence cleanupFence{VK_NULL_HANDLE};
      vk::Semaphore presentSemaphore{VK_NULL_HANDLE};
      std::vector<SwapchainCleanupData> oldSwapchains;
      u32 imageIndex{s_invalid_image_index};
    };
    // entries for old swapchains, imageIndex = s_invalid_image_index
    // otherwise for current valid swapchain
    std::deque<PresentOperationInfo> presentHistory;
    void cleanupPresentHistory();
    void cleanupPresentInfo(PresentOperationInfo &presentInfo);
    void cleanupOldSwapchain(SwapchainCleanupData &oldSwapchain);

    /// cleanup data (image-related, swapchain + semaphore)
    std::array<std::vector<SwapchainObjects>, num_buffered_frames> swapchainGarbages;  // indexed by frameIndex
    std::vector<SwapchainObjects> &currentSwapchainGarbage() noexcept {
      return swapchainGarbages[frameIndex];
    }
    void cleanupSwapchainObjects(
        SwapchainObjects &garbage);  // swapchainObjects -> swapchainGarbages -> cleanup

    // 1. presentHistory -> oldSwapchains -> cleanup (swapchain recreation)
    // 2. presentHistory (associated with signaled fence) -> cleanup (upon queue present)
    std::vector<SwapchainCleanupData> oldSwapchains;
    void scheduleOldSwapchainForDestruction(vk::SwapchainKHR oldSwapchain);

    int frameIndex{-1};
  };

  // ref: LegitEngine (https://github.com/Raikiri/LegitEngine), nvpro_core
  struct ZPC_CORE_API SwapchainBuilder {
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
    SwapchainBuilder &setSamples(vk::SampleCountFlagBits numSamples) {
      sampleBits = numSamples;
      return *this;
    }

    Swapchain build(vk::RenderPass renderPass = VK_NULL_HANDLE) {
      Swapchain obj(ctx);
      build(obj, renderPass);
      return obj;
    }
    void resize(Swapchain &obj, u32 width, u32 height, vk::RenderPass renderPass = VK_NULL_HANDLE);

  private:
    void build(Swapchain &obj, vk::RenderPass renderPass = VK_NULL_HANDLE);

    VulkanContext &ctx;
    vk::SurfaceKHR surface;
    std::vector<vk::SurfaceFormatKHR> surfFormats;
    std::vector<vk::PresentModeKHR> surfPresentModes;
    vk::SurfaceCapabilitiesKHR surfCapabilities;
    vk::Format swapchainDepthFormat;
    vk::SampleCountFlagBits sampleBits;

    vk::SwapchainCreateInfoKHR ci;
    bool buildDepthBuffer{false};
  };

}  // namespace zs