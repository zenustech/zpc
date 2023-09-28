#pragma once
#include <memory>
#include <string>
#include <vector>
//
#include "vulkan/vulkan.hpp"

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "vma/vk_mem_alloc.h"
//
#include "zensim/vulkan/VkUtils.hpp"
#include "zensim/zpc_tpls/fmt/format.h"

#define ZS_VULKAN_USE_VMA 1

namespace zs {

  struct Image;
  struct ImageView;
  struct Buffer;
  struct Framebuffer;
  struct RenderPass;
  struct Swapchain;
  struct SwapchainBuilder;
  struct ShaderModule;
  struct Pipeline;
  struct PipelineBuilder;
  struct DescriptorSetLayoutBuilder;
  struct DescriptorPool;
  struct ExecutionContext;

  struct Vulkan;

  /// @note CAUTION: must match the member order defined in VulkanContext
  enum vk_queue_e { graphics = 0, compute, transfer };
  enum vk_cmd_usage_e { reuse = 0, single_use, reset };

  struct VulkanContext {
    Vulkan &driver() const noexcept;
    VulkanContext(int devid, vk::Instance instance, vk::PhysicalDevice device,
                  const vk::DispatchLoaderDynamic &instDispatcher);
    ~VulkanContext() noexcept = default;
    VulkanContext(VulkanContext &&) = default;
    VulkanContext &operator=(VulkanContext &&) = default;
    VulkanContext(const VulkanContext &) = delete;
    VulkanContext &operator=(const VulkanContext &) = delete;

    auto getDevId() const noexcept { return devid; }

    /// queries
    u32 numDistinctQueueFamilies() const noexcept { return uniqueQueueFamilyIndices.size(); }

    vk::PhysicalDevice getPhysicalDevice() const noexcept { return physicalDevice; }
    vk::Device getDevice() const noexcept { return device; }
    int getQueueFamilyIndex(vk_queue_e e = vk_queue_e::graphics) const noexcept {
      return queueFamilyIndices[e];
    }
    bool retrieveQueue(vk::Queue &q, vk_queue_e e = vk_queue_e::graphics,
                       u32 i = 0) const noexcept {
      auto index = queueFamilyIndices[e];
      if (index != -1) {
        q = device.getQueue(index, i, dispatcher);
        return true;
      }
      return false;
    }
    vk::Queue getQueue(vk_queue_e e = vk_queue_e::graphics, u32 i = 0) const {
      auto index = queueFamilyIndices[e];
      if (index == -1) throw std::runtime_error("queue does not exist.");
      return device.getQueue(index, i, dispatcher);
    }
    vk::DescriptorPool descriptorPool() const noexcept { return defaultDescriptorPool; }
    VmaAllocator &allocator() noexcept { return defaultAllocator; }
    const VmaAllocator &allocator() const noexcept { return defaultAllocator; }

    bool supportGraphics() const { return graphicsQueueFamilyIndex != -1; }
    /// @note usually called right before swapchain creation for assurance
    bool supportSurface(vk::SurfaceKHR surface) const {
      if (graphicsQueueFamilyIndex == -1) return false;
      return physicalDevice.getSurfaceSupportKHR(graphicsQueueFamilyIndex, surface, dispatcher);
    }
    u32 numMemoryTypes() const { return memoryProperties.memoryTypeCount; }
    u32 findMemoryType(u32 memoryTypeBits, vk::MemoryPropertyFlags properties) {
      for (u32 i = 0; i < memoryProperties.memoryTypeCount; i++)
        if ((memoryTypeBits & (1 << i))
            && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
          return i;
        }
      throw std::runtime_error(
          fmt::format("Failed to find a suitable memory type (within {:b} typebits) satisfying "
                      "the property flag [{:0>10b}]!\n",
                      memoryTypeBits, get_flag_value(properties)));
    }
    vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates,
                                   vk::ImageTiling tiling, vk::FormatFeatureFlags features) const {
      for (vk::Format format : candidates) {
        VkFormatProperties props;
        dispatcher.vkGetPhysicalDeviceFormatProperties(physicalDevice,
                                                       static_cast<VkFormat>(format), &props);

        if (tiling == vk::ImageTiling::eLinear
            && (vk::FormatFeatureFlags{props.linearTilingFeatures} & features) == features) {
          return format;
        } else if (tiling == vk::ImageTiling::eOptimal
                   && (vk::FormatFeatureFlags{props.optimalTilingFeatures} & features)
                          == features) {
          return format;
        }
      }
      throw std::runtime_error(fmt::format(
          "cannot find a suitable candidate (among {}) format that supports [{}] "
          "tiling and has [{}] features",
          candidates.size(), static_cast<std::underlying_type_t<vk::ImageTiling>>(tiling),
          get_flag_value(features)));
    }

    /// behaviors
    void reset();
    void sync() const { device.waitIdle(dispatcher); }

    /// resource builders
    void setupDefaultDescriptorPool();
    // should not delete this then acquire again for same usage
    void acquireSet(vk::DescriptorSetLayout descriptorSetLayout, vk::DescriptorSet &set) const {
      set = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}
                                              .setDescriptorPool(defaultDescriptorPool)
                                              .setPSetLayouts(&descriptorSetLayout)
                                              .setDescriptorSetCount(1))[0];
      /// @note from lve
      // Might want to create a "DescriptorPoolManager" class that handles this case, and builds
      // a new pool whenever an old pool fills up. But this is beyond our current scope
    }
    SwapchainBuilder &swapchain(vk::SurfaceKHR surface, bool reset = false);
    PipelineBuilder pipeline();
    DescriptorSetLayoutBuilder setlayout();
    ExecutionContext &env();  // thread-safe

    Buffer createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                        vk::MemoryPropertyFlags props = vk::MemoryPropertyFlagBits::eDeviceLocal);
    Buffer createStagingBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage);

    Image createImage(vk::ImageCreateInfo imageCI,
                      vk::MemoryPropertyFlags props = vk::MemoryPropertyFlagBits::eDeviceLocal,
                      bool createView = true);
    Image create2DImage(const vk::Extent2D &dim, vk::Format format = vk::Format::eR8G8B8A8Unorm,
                        vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled,
                        vk::MemoryPropertyFlags props = vk::MemoryPropertyFlagBits::eDeviceLocal,
                        bool mipmaps = false, bool createView = true);

    ImageView create2DImageView(vk::Image image, vk::Format format = vk::Format::eR8G8B8A8Unorm,
                                vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor,
                                u32 levels = VK_REMAINING_MIP_LEVELS,
                                const void *pNextImageView = nullptr);

    Framebuffer createFramebuffer(const std::vector<vk::ImageView> &imageViews, vk::Extent2D size,
                                  vk::RenderPass renderPass);
    DescriptorPool createDescriptorPool(const std::vector<vk::DescriptorPoolSize> &poolSizes,
                                        u32 maxSets = 1000);
    ShaderModule createShaderModule(const std::vector<char> &code,
                                    vk::ShaderStageFlagBits stageFlag);
    ShaderModule createShaderModule(const u32 *code, size_t size,
                                    vk::ShaderStageFlagBits stageFlag);

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
    vk::PhysicalDeviceMemoryProperties memoryProperties;
    vk::DescriptorPool defaultDescriptorPool;
    VmaAllocator defaultAllocator;

  protected:
    friend struct VkPipeline;

    /// resource builders
    // generally at most one swapchain is associated with a context, thus reuse preferred
    std::unique_ptr<SwapchainBuilder> swapchainBuilder;
  };

  struct ExecutionContext {
    ExecutionContext(VulkanContext &ctx);
    ~ExecutionContext();

    struct PoolFamily {
      vk::CommandPool reusePool;      // submit multiple times
      vk::CommandPool singleUsePool;  // submit once
      vk::CommandPool resetPool;      // reset and re-record
      vk::Queue queue;
      VulkanContext *pctx{nullptr};

      vk::CommandPool cmdpool(vk_cmd_usage_e usage = vk_cmd_usage_e::reset) {
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

      vk::CommandBuffer createCommandBuffer(vk::CommandBufferLevel level
                                            = vk::CommandBufferLevel::ePrimary,
                                            bool begin = true,
                                            const vk::CommandBufferInheritanceInfo *pInheritanceInfo
                                            = nullptr,
                                            vk_cmd_usage_e usage = vk_cmd_usage_e::single_use) {
        auto cmdPool = cmdpool(usage);
        vk::CommandBufferUsageFlags usageFlags{};
        if (usage == vk_cmd_usage_e::single_use || usage == vk_cmd_usage_e::reset)
          usageFlags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        else
          usageFlags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;

        std::vector<vk::CommandBuffer> cmd = pctx->device.allocateCommandBuffers(
            vk::CommandBufferAllocateInfo{cmdPool, level, (u32)1}, pctx->dispatcher);

        // if (usage == vk_cmd_usage_e::reset) cmds.push_back(cmd[0]);

        if (begin) cmd[0].begin(vk::CommandBufferBeginInfo{usageFlags, pInheritanceInfo});

        return cmd[0];
      }
      void submit(u32 count, const vk::CommandBuffer *cmds, vk::Fence fence,
                  vk_cmd_usage_e usage = vk_cmd_usage_e::single_use) {
        for (u32 i = 0; i < count; i++) cmds[i].end();

        vk::SubmitInfo submit{};
        submit.setCommandBufferCount(count).setPCommandBuffers(cmds);
        if (auto res = queue.submit(1, &submit, fence, pctx->dispatcher);
            res != vk::Result::eSuccess)
          throw std::runtime_error(fmt::format("failed to submit {} commands to queue.", count));
        if (usage == vk_cmd_usage_e::single_use)
          pctx->device.freeCommandBuffers(singleUsePool, count, cmds, pctx->dispatcher);
      }

      void submit(const vk::CommandBuffer &cmd, vk::Fence fence,
                  vk_cmd_usage_e usage = vk_cmd_usage_e::single_use) {
        submit(1, &cmd, fence, usage);
      }
    };

    PoolFamily &pools(vk_queue_e e = vk_queue_e::graphics) {
      return poolFamilies[ctx.queueFamilyMaps[e]];
    }
    void resetCmds(vk_cmd_usage_e usage, vk_queue_e e = vk_queue_e::graphics) {
      ctx.device.resetCommandPool(pools(e).cmdpool(usage), {}, ctx.dispatcher);
    }

    std::vector<PoolFamily> poolFamilies;

  protected:
    VulkanContext &ctx;
  };

  u32 check_current_working_contexts();

}  // namespace zs