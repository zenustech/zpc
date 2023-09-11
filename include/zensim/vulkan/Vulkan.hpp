#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>
//
#include "vulkan/vulkan.hpp"
//
#include "zensim/Reflection.h"
#include "zensim/Singleton.h"
#include "zensim/profile/CppTimers.hpp"
#include "zensim/types/SourceLocation.hpp"
#include "zensim/types/Tuple.h"
#include "zensim/zpc_tpls/fmt/format.h"

namespace zs {

  struct SwapchainBuilder;

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

      int devid;
      vk::PhysicalDevice physicalDevice;
      vk::Device device;                     // currently dedicated for rendering
      vk::DispatchLoaderDynamic dispatcher;  // store device-specific calls
      int graphicsQueueFamilyIndex;
      vk::Queue queue;

    protected:
      /// resource builders
      // generally at most one swapchain is associated with a context, thus reuse preferred
      std::unique_ptr<SwapchainBuilder> swapchainBuilder;
    };

  private:
    vk::Instance _instance;
    vk::DispatchLoaderDynamic _dispatcher;  // store vulkan-instance calls
    vk::DebugUtilsMessengerEXT _messenger;
    std::vector<VulkanContext> _contexts;  ///< generally one per device
    int _defaultContext;
  };

  struct Buffer {
    Buffer(Vulkan::VulkanContext &ctx) : ctx{ctx}, buffer{} {}
    ~Buffer() { ctx.device.destroyBuffer(buffer, nullptr, ctx.dispatcher); }
    vk::Buffer operator*() const { return buffer; }
    operator vk::Buffer() const { return buffer; }

  protected:
    Vulkan::VulkanContext &ctx;
    vk::Buffer buffer;
  };
  struct BufferView {
    BufferView(Vulkan::VulkanContext &ctx) : ctx{ctx}, bufv{} {}
    ~BufferView() { ctx.device.destroyBufferView(bufv, nullptr, ctx.dispatcher); }
    vk::BufferView operator*() const { return bufv; }
    operator vk::BufferView() const { return bufv; }

  protected:
    Vulkan::VulkanContext &ctx;
    vk::BufferView bufv;
  };

  struct Image {
    Image(Vulkan::VulkanContext &ctx) : ctx{ctx}, image{} {}
    ~Image() { ctx.device.destroyImage(image, nullptr, ctx.dispatcher); }
    vk::Image operator*() const { return image; }
    operator vk::Image() const { return image; }

  protected:
    Vulkan::VulkanContext &ctx;
    vk::Image image;
  };
  struct ImageBuilder {
    ;
  };
  struct ImageView {
    ImageView(Vulkan::VulkanContext &ctx) : ctx{ctx}, imgv{} {}
    ~ImageView() { ctx.device.destroyImageView(imgv, nullptr, ctx.dispatcher); }
    vk::ImageView operator*() const { return imgv; }
    operator vk::ImageView() const { return imgv; }

  protected:
    Vulkan::VulkanContext &ctx;
    vk::ImageView imgv;
  };
  struct ImageViewBuilder {
    ;
  };

  struct Swapchain {
    Swapchain(Vulkan::VulkanContext &ctx) : ctx{ctx}, swapchain{} {}
    Swapchain(const Swapchain &) = delete;
    Swapchain(Swapchain &&) noexcept = default;
    ~Swapchain() { ctx.device.destroySwapchainKHR(swapchain, nullptr, ctx.dispatcher); }

    std::vector<vk::Image> images() const {
      return ctx.device.getSwapchainImagesKHR(swapchain, ctx.dispatcher);
    }

    // update width, height
    vk::SwapchainKHR operator*() const { return swapchain; }
    operator vk::SwapchainKHR() const { return swapchain; }

  protected:
    friend struct SwapchainBuilder;

    Vulkan::VulkanContext &ctx;
    vk::SwapchainKHR swapchain;
  };

  // ref: LegitEngine (https://github.com/Raikiri/LegitEngine), nvpro_core
  struct SwapchainBuilder {
    SwapchainBuilder(Vulkan::VulkanContext &ctx, vk::SurfaceKHR targetSurface);
    SwapchainBuilder(const SwapchainBuilder &) = delete;
    SwapchainBuilder(SwapchainBuilder &&) noexcept = default;

    vk::SurfaceKHR getSurface() const { return surface; }

    void imageCount(u32 count) {
      ci.minImageCount = std::clamp(count, (u32)surfCapabilities.minImageCount,
                                    (u32)surfCapabilities.maxImageCount);
    }
    void extent(u32 width, u32 height) { ci.imageExtent = vk::Extent2D{width, height}; }
    void presentMode(vk::PresentModeKHR mode);
    void usage(vk::ImageUsageFlags flags) { ci.imageUsage = flags; }

    Swapchain build() {
      Swapchain obj(ctx);
      obj.swapchain = ctx.device.createSwapchainKHR(ci, nullptr, ctx.dispatcher);
      return obj;
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
