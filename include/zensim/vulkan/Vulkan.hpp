#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
//
#include "vulkan/vulkan.hpp"
//
#include "zensim/Reflection.h"
#include "zensim/Singleton.h"
#include "zensim/execution/ConcurrencyPrimitive.hpp"
#include "zensim/profile/CppTimers.hpp"
#include "zensim/types/SourceLocation.hpp"
#include "zensim/types/Tuple.h"
#include "zensim/zpc_tpls/fmt/format.h"

namespace zs {

  struct SwapchainBuilder;
  struct ExecutionContext;

  /// @note CAUTION: must match the member order defined in VulkanContext
  enum vk_queue_e { graphics = 0, compute, transfer };
  enum vk_cmd_usage_e { reuse = 0, single_use, reset };

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

    /// begin vulkan context
    struct VulkanContext {
      auto &driver() const noexcept { return Vulkan::driver(); }
      VulkanContext(int devid, vk::PhysicalDevice device,
                    const vk::DispatchLoaderDynamic &instDispatcher);
      ~VulkanContext() noexcept = default;
      VulkanContext(VulkanContext &&) = default;
      VulkanContext &operator=(VulkanContext &&) = default;
      VulkanContext(const VulkanContext &) = delete;
      VulkanContext &operator=(const VulkanContext &) = delete;

      auto getDevId() const noexcept { return devid; }

      /// queries
      u32 numDistinctQueueFamilies() const noexcept { return uniqueQueueFamilyIndices.size(); }

      bool retrieveQueue(vk::Queue &q, vk_queue_e e = vk_queue_e::graphics,
                         u32 i = 0) const noexcept {
        auto index = queueFamilyIndices[e];
        if (index != -1) {
          q = device.getQueue(index, i, dispatcher);
          return true;
        }
        return false;
      }
      vk::Queue getComputeQueue(vk_queue_e e = vk_queue_e::graphics, u32 i = 0) const {
        auto index = queueFamilyIndices[e];
        if (index != -1) throw std::runtime_error("compute queue does not exist.");
        return device.getQueue(index, i, dispatcher);
      }

      bool supportGraphics() const { return graphicsQueueFamilyIndex != -1; }
      /// @note usually called right before swapchain creation for assurance
      bool supportSurface(vk::SurfaceKHR surface) const {
        if (graphicsQueueFamilyIndex == -1) return false;
        return physicalDevice.getSurfaceSupportKHR(graphicsQueueFamilyIndex, surface, dispatcher);
      }

      /// behaviors
      void reset();
      void sync() const { device.waitIdle(dispatcher); }

      /// resource builders
      inline SwapchainBuilder &swapchain(vk::SurfaceKHR surface, bool reset = false);
      ExecutionContext &env();  // thread-safe

      int devid;
      vk::PhysicalDevice physicalDevice;
      vk::Device device;                     // currently dedicated for rendering
      vk::DispatchLoaderDynamic dispatcher;  // store device-specific calls
      // graphics queue family should also be used for presentation if swapchain required
      union {
        int queueFamilyIndices[3];
        int graphicsQueueFamilyIndex, computeQueueFamilyIndex, transferQueueFamilyIndex;
      };
      union {
        int queueFamilyMaps[3];
        int graphicsQueueFamilyMap, computeQueueFamilyMap, transferQueueFamilyMap;
      };
      std::vector<u32> uniqueQueueFamilyIndices;

    protected:
      /// resource builders
      // generally at most one swapchain is associated with a context, thus reuse preferred
      std::unique_ptr<SwapchainBuilder> swapchainBuilder;
    };
    /// end vulkan context

  private:
    vk::Instance _instance;
    vk::DispatchLoaderDynamic _dispatcher;  // store vulkan-instance calls
    vk::DebugUtilsMessengerEXT _messenger;
    std::vector<VulkanContext> _contexts;  ///< generally one per device
    int _defaultContext = 0;
  };

  u32 check_current_working_contexts();

  struct ExecutionContext {
    ExecutionContext(Vulkan::VulkanContext &ctx);
    ~ExecutionContext() {
      for (auto &family : poolFamilies) {
        ctx.device.resetCommandPool(
            family.reusePool, vk::CommandPoolResetFlagBits::eReleaseResources, ctx.dispatcher);
        ctx.device.destroyCommandPool(family.reusePool, nullptr, ctx.dispatcher);

        ctx.device.resetCommandPool(
            family.singleUsePool, vk::CommandPoolResetFlagBits::eReleaseResources, ctx.dispatcher);
        ctx.device.destroyCommandPool(family.singleUsePool, nullptr, ctx.dispatcher);

        ctx.device.resetCommandPool(
            family.resetPool, vk::CommandPoolResetFlagBits::eReleaseResources, ctx.dispatcher);
        ctx.device.destroyCommandPool(family.resetPool, nullptr, ctx.dispatcher);
      }
    }

    struct PoolFamily {
      vk::CommandPool reusePool;      // submit multiple times
      vk::CommandPool singleUsePool;  // submit once
      vk::CommandPool resetPool;      // reset and re-record

      vk::CommandPool pool(vk_cmd_usage_e usage = vk_cmd_usage_e::reset) {
        switch (usage) {
          case vk_cmd_usage_e::reuse:
            return reusePool;
          case vk_cmd_usage_e::single_use:
            return singleUsePool;
          case vk_cmd_usage_e::reset:
            return resetPool;
          default:
            return resetPool;
        }
      }
    };

    PoolFamily &pools(vk_queue_e e = vk_queue_e::graphics) {
      return poolFamilies[ctx.queueFamilyMaps[e]];
    }
    void resetCmds(vk_cmd_usage_e usage, vk_queue_e e = vk_queue_e::graphics) {
      ctx.device.resetCommandPool(pools(e).pool(usage), {}, ctx.dispatcher);
    }

    std::vector<PoolFamily> poolFamilies;

  protected:
    Vulkan::VulkanContext &ctx;
  };

  struct Swapchain {
    Swapchain() = delete;
    Swapchain(Vulkan::VulkanContext &ctx) : ctx{ctx}, swapchain{} {}
    Swapchain(const Swapchain &) = delete;
    Swapchain(Swapchain &&) noexcept = default;
    ~Swapchain() {
      resetAux();
      ctx.device.destroySwapchainKHR(swapchain, nullptr, ctx.dispatcher);
    }

    std::vector<vk::Image> getImages() const { return images; }
    u32 imageCount() const { return images.size(); }
    u32 acquireNextImage() {
      if (vk::Result res
          = ctx.device.waitForFences(1, &readFences[frameIndex], VK_TRUE,
                                     detail::deduce_numeric_max<u64>(), ctx.dispatcher);
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
    void nextFrame() { frameIndex = (frameIndex + 1) % imageCount(); }

    // update width, height
    vk::SwapchainKHR operator*() const { return swapchain; }
    operator vk::SwapchainKHR() const { return swapchain; }

  protected:
    void resetAux() {
      for (auto &v : imageViews) ctx.device.destroyImageView(v, nullptr, ctx.dispatcher);
      for (auto &s : readSemaphores) ctx.device.destroySemaphore(s, nullptr, ctx.dispatcher);
      for (auto &s : writeSemaphores) ctx.device.destroySemaphore(s, nullptr, ctx.dispatcher);
      for (auto &f : readFences) ctx.device.destroyFence(f, nullptr, ctx.dispatcher);
      for (auto &f : writeFences) ctx.device.destroyFence(f, nullptr, ctx.dispatcher);
    }
    friend struct SwapchainBuilder;

    Vulkan::VulkanContext &ctx;
    vk::SwapchainKHR swapchain;
    std::vector<vk::Image> images;
    std::vector<vk::ImageView> imageViews;
    std::vector<vk::Semaphore> readSemaphores;
    std::vector<vk::Semaphore> writeSemaphores;
    // littleVulkanEngine-alike setup
    std::vector<vk::Fence> readFences;
    std::vector<vk::Fence> writeFences;
    int frameIndex;
  };

  // ref: LegitEngine (https://github.com/Raikiri/LegitEngine), nvpro_core
  struct SwapchainBuilder {
    SwapchainBuilder(Vulkan::VulkanContext &ctx, vk::SurfaceKHR targetSurface);
    SwapchainBuilder(const SwapchainBuilder &) = delete;
    SwapchainBuilder(SwapchainBuilder &&) noexcept = default;
    ~SwapchainBuilder() {
      zs::Vulkan::vk_inst().destroySurfaceKHR(surface, nullptr, zs::Vulkan::vk_inst_dispatcher());
    }

    vk::SurfaceKHR getSurface() const { return surface; }

    void imageCount(u32 count) {
      ci.minImageCount = std::clamp(count, (u32)surfCapabilities.minImageCount,
                                    (u32)surfCapabilities.maxImageCount);
    }
    void extent(u32 width, u32 height) { ci.imageExtent = vk::Extent2D{width, height}; }
    void presentMode(vk::PresentModeKHR mode = vk::PresentModeKHR::eMailbox);
    void usage(vk::ImageUsageFlags flags) { ci.imageUsage = flags; }

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
    Vulkan::VulkanContext &ctx;
    vk::SurfaceKHR surface;
    std::vector<vk::SurfaceFormatKHR> surfFormats;
    std::vector<vk::PresentModeKHR> surfPresentModes;
    vk::SurfaceCapabilitiesKHR surfCapabilities;

    vk::SwapchainCreateInfoKHR ci;
  };

  SwapchainBuilder &Vulkan::VulkanContext::swapchain(vk::SurfaceKHR surface, bool reset) {
    if (!swapchainBuilder || reset || swapchainBuilder->getSurface() != surface)
      swapchainBuilder.reset(new SwapchainBuilder(*this, surface));
    return *swapchainBuilder;
  }

}  // namespace zs
