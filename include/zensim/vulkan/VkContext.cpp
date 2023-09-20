#include "zensim/vulkan/VkContext.hpp"

#include <iostream>
#include <map>
#include <set>
#include <thread>

#include "zensim/vulkan/VkBuffer.hpp"
#include "zensim/vulkan/VkDescriptor.hpp"
#include "zensim/vulkan/VkImage.hpp"
#include "zensim/vulkan/VkPipeline.hpp"
#include "zensim/vulkan/VkRenderPass.hpp"
#include "zensim/vulkan/VkShader.hpp"
#include "zensim/vulkan/VkSwapchain.hpp"
#include "zensim/vulkan/Vulkan.hpp"

//
#include "zensim/Logger.hpp"
#include "zensim/Platform.hpp"
#include "zensim/ZpcReflection.hpp"
#include "zensim/execution/ConcurrencyPrimitive.hpp"
#include "zensim/types/Iterator.h"
#include "zensim/types/SourceLocation.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

namespace zs {

  using ContextEnvs = std::map<int, ExecutionContext>;
  using WorkerEnvs = std::map<std::thread::id, ContextEnvs>;
  namespace {
    static WorkerEnvs g_workingContexts;
    static Mutex g_mtx{};
  }  // namespace

  ///
  ///
  /// vulkan context
  ///
  ///
  Vulkan& VulkanContext::driver() const noexcept { return Vulkan::driver(); }

  void VulkanContext::reset() {
    /// clear builders
    // if (swapchainBuilder) swapchainBuilder.reset(nullptr);
    /// clear resources
    {
      // working contexts (command pool resources)
      // g_mtx.lock();
      if (g_mtx.try_lock()) {
        g_workingContexts.clear();
        g_mtx.unlock();
      } else
        throw std::runtime_error(
            "Other worker threads are still accessing vk command contexts while the ctx is being "
            "destroyed!");
      // g_mtx.unlock();
    }
    {
      // descriptor pool resources
      device.resetDescriptorPool(defaultDescriptorPool, vk::DescriptorPoolResetFlags{}, dispatcher);
      device.destroyDescriptorPool(defaultDescriptorPool, nullptr, dispatcher);
      defaultDescriptorPool = VK_NULL_HANDLE;
    }
    /// destroy logical device
    device.destroy(nullptr, dispatcher);
    fmt::print("vulkan context [{}] (of {}) has been successfully reset.\n", devid,
               driver().num_devices());
  }

  VulkanContext::VulkanContext(int devId, vk::PhysicalDevice phydev,
                               const vk::DispatchLoaderDynamic& instDispatcher)
      : devid{devId}, physicalDevice{phydev}, device{}, dispatcher{instDispatcher} {
    /// @note logical device
    std::vector<vk::ExtensionProperties> devExts
        = physicalDevice.enumerateDeviceExtensionProperties();
    vk::PhysicalDeviceProperties devProps = physicalDevice.getProperties();

    /// queue family
    auto queueFamilyProps = physicalDevice.getQueueFamilyProperties();
    // graphicsQueueFamilyIndex = computeQueueFamilyIndex = transferQueueFamilyIndex = -1;
    for (auto& queueFamilyIndex : queueFamilyIndices) queueFamilyIndex = -1;
    for (auto& queueFamilyMap : queueFamilyMaps) queueFamilyMap = -1;
    for (int i = 0; i != queueFamilyProps.size(); ++i) {
      auto& q = queueFamilyProps[i];
      if (graphicsQueueFamilyIndex == -1 && (q.queueFlags & vk::QueueFlagBits::eGraphics)) {
        graphicsQueueFamilyIndex = i;
        ZS_WARN_IF(!(q.queueFlags & vk::QueueFlagBits::eTransfer),
                   "the selected graphics queue family cannot transfer!");
      }
      if (computeQueueFamilyIndex == -1 && (q.queueFlags & vk::QueueFlagBits::eCompute))
        computeQueueFamilyIndex = i;
      if (transferQueueFamilyIndex == -1 && (q.queueFlags & vk::QueueFlagBits::eTransfer))
        transferQueueFamilyIndex = i;
    }
    ZS_ERROR_IF(graphicsQueueFamilyIndex == -1, "graphics queue family does not exist!");
    fmt::print("selected queue family [{}] for graphics!\n", graphicsQueueFamilyIndex);

    std::set<u32> uniqueQueueFamilyIndices{
        (u32)graphicsQueueFamilyIndex, (u32)computeQueueFamilyIndex, (u32)transferQueueFamilyIndex};
    this->uniqueQueueFamilyIndices.reserve(uniqueQueueFamilyIndices.size());
    std::vector<vk::DeviceQueueCreateInfo> dqCIs(uniqueQueueFamilyIndices.size());
    float priority = 1.f;
    {
      int i;
      for (auto index : uniqueQueueFamilyIndices) {
        auto& dqCI = dqCIs[i];
        this->uniqueQueueFamilyIndices.push_back(index);
        dqCI.setQueueFamilyIndex(index).setQueueCount(1).setPQueuePriorities(&priority);

        if (graphicsQueueFamilyIndex == index) graphicsQueueFamilyMap = i;
        if (computeQueueFamilyIndex == index) computeQueueFamilyMap = i;
        if (transferQueueFamilyIndex == index) transferQueueFamilyMap = i;
        i++;
      }
    }

    /// extensions
    int rtPreds = 0;
    constexpr int rtRequiredPreds = 5;
    /// @note the first 5 extensions are required for rt support
    std::vector<const char*> expectedExtensions{
        "VK_KHR_ray_tracing_pipeline",     "VK_KHR_acceleration_structure",
        "VK_EXT_descriptor_indexing",      "VK_KHR_buffer_device_address",
        "VK_KHR_deferred_host_operations", "VK_KHR_swapchain"};
    std::vector<const char*> enabledExtensions(0);
    // pick up supported extensions
    for (int i = 0; i != expectedExtensions.size(); ++i) {
      auto ext = expectedExtensions[i];
      for (auto& devExt : devExts) {
        if (strcmp(ext, devExt.extensionName) == 0) {
          enabledExtensions.push_back(ext);
          if (i < rtRequiredPreds) rtPreds++;
          break;
        }
      }
    }
    vk::DeviceCreateInfo devCI{{},
                               (u32)dqCIs.size(),
                               dqCIs.data(),
                               0,
                               nullptr,
                               (u32)enabledExtensions.size(),
                               enabledExtensions.data()};

    /// features
    // ref: TU Wien Vulkan Tutorial Ep1
    vk::PhysicalDeviceVulkan12Features vk12Features{};
    vk12Features.descriptorIndexing = VK_TRUE;
    vk12Features.bufferDeviceAddress = VK_TRUE;
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
    asFeatures.accelerationStructure = VK_TRUE;
    asFeatures.pNext = &vk12Features;
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rtPipeFeatures{};
    rtPipeFeatures.rayTracingPipeline = VK_TRUE;
    rtPipeFeatures.pNext = &asFeatures;
    if (rtPreds == rtRequiredPreds) {
      devCI.pNext = &rtPipeFeatures;
    }

    device = physicalDevice.createDevice(devCI, nullptr, dispatcher);
    dispatcher.init(device);
    ZS_ERROR_IF(!device, fmt::format("Vulkan device [{}] failed initialization!\n", devid));

    VkPhysicalDeviceMemoryProperties tmp;
    dispatcher.vkGetPhysicalDeviceMemoryProperties(physicalDevice, &tmp);
    memoryProperties = tmp;

    setupDefaultDescriptorPool();

    /// display info
    fmt::print(
        "\t[InitInfo -- Dev Property] Vulkan device [{}] name: {}."
        "\n\t\t(Graphics/Compute/Transfer) queue family index: {}, {}, {}. Ray-tracing support: "
        "{}. "
        "\n\tEnabled the following device tensions ({} in total):",
        devid, devProps.deviceName, graphicsQueueFamilyIndex, computeQueueFamilyIndex,
        transferQueueFamilyIndex, rtPreds == rtRequiredPreds, enabledExtensions.size());
    u32 accum = 0;
    for (auto ext : enabledExtensions) {
      if ((accum++) % 2 == 0) fmt::print("\n\t\t");
      fmt::print("{}\t", ext);
    }
    fmt::print("\n\tManaging the following [{}] memory type(s) in total:\n",
               memoryProperties.memoryTypeCount);
    for (u32 typeIndex = 0; typeIndex < memoryProperties.memoryTypeCount; ++typeIndex) {
      auto propertyFlags = memoryProperties.memoryTypes[typeIndex].propertyFlags;
      using BitType = typename RM_REF_T(propertyFlags)::MaskType;
      std::string tag;
      if (propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) tag += "device_local; ";
      if (propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent) tag += "host_coherent; ";
      if (propertyFlags & vk::MemoryPropertyFlagBits::eHostCached) tag += "host_cached; ";
      if (propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible) tag += "host_visible; ";
      if (propertyFlags & vk::MemoryPropertyFlagBits::eProtected) tag += "protected; ";
      if (propertyFlags & vk::MemoryPropertyFlagBits::eLazilyAllocated) tag += "lazily_allocated; ";
      tag += "...";
      fmt::print("\t\t[{}] flag:\t{:0>10b} ({})\n", typeIndex, static_cast<BitType>(propertyFlags),
                 tag);
    }
  }

  void VulkanContext::setupDefaultDescriptorPool() {
    std::vector<vk::DescriptorPoolSize> poolSizes;
    auto uniformPoolSize = vk::DescriptorPoolSize().setDescriptorCount(1000).setType(
        vk::DescriptorType::eUniformBufferDynamic);
    poolSizes.push_back(uniformPoolSize);

    auto imageSamplerPoolSize = vk::DescriptorPoolSize().setDescriptorCount(1000).setType(
        vk::DescriptorType::eSampledImage);  // eCombinedImageSampler
    poolSizes.push_back(imageSamplerPoolSize);

    auto storagePoolSize = vk::DescriptorPoolSize().setDescriptorCount(1000).setType(
        vk::DescriptorType::eStorageBuffer);
    poolSizes.push_back(storagePoolSize);

    auto storageImagePoolSize = vk::DescriptorPoolSize().setDescriptorCount(1000).setType(
        vk::DescriptorType::eStorageImage);
    poolSizes.push_back(storageImagePoolSize);

    defaultDescriptorPool = device.createDescriptorPool(
        vk::DescriptorPoolCreateInfo{}
            .setPoolSizeCount(poolSizes.size())
            .setPPoolSizes(poolSizes.data())
            .setMaxSets(1000)
            .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet),
        nullptr, dispatcher);
  }

  ExecutionContext& VulkanContext::env() {
    WorkerEnvs::iterator workerIter;
    ContextEnvs::iterator iter;
    g_mtx.lock();
    bool tag;
    std::tie(workerIter, tag)
        = g_workingContexts.emplace(std::this_thread::get_id(), ContextEnvs{});
    std::tie(iter, tag) = workerIter->second.emplace(devid, *this);
    g_mtx.unlock();
    return iter->second;
  }
  u32 check_current_working_contexts() { return g_workingContexts.size(); }

  ///
  /// builders
  ///
  SwapchainBuilder& VulkanContext::swapchain(vk::SurfaceKHR surface, bool reset) {
    if (!swapchainBuilder || reset || swapchainBuilder->getSurface() != surface)
      swapchainBuilder.reset(new SwapchainBuilder(*this, surface));
    return *swapchainBuilder;
  }
  PipelineBuilder VulkanContext::pipeline() { return PipelineBuilder{*this}; }
  DescriptorSetLayoutBuilder VulkanContext::setlayout() {
    return DescriptorSetLayoutBuilder{*this};
  }

  Buffer VulkanContext::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                     vk::MemoryPropertyFlags props) {
    Buffer buffer(*this);

    vk::BufferCreateInfo bufCI{};
    bufCI.setUsage(usage);
    bufCI.setSize(size);
    bufCI.setSharingMode(vk::SharingMode::eExclusive);
    auto buf = device.createBuffer(bufCI, nullptr, dispatcher);

    vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buf, dispatcher);
    u32 memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, props);
    vk::MemoryAllocateInfo allocInfo{memRequirements.size, memoryTypeIndex};
    auto mem = device.allocateMemory(allocInfo, nullptr, dispatcher);

    device.bindBufferMemory(buf, mem, 0, dispatcher);

    VkMemory memory{*this};
    memory.mem = mem;
    memory.memSize = memRequirements.size;
    memory.memoryPropertyFlags = memoryProperties.memoryTypes[memoryTypeIndex].propertyFlags;

    buffer.size = size;
    buffer.usageFlags = usage;
    buffer.alignment = memRequirements.alignment;
    buffer.buffer = buf;
    buffer.pmem = std::make_shared<VkMemory>(std::move(memory));
    return buffer;
  }
  Buffer VulkanContext::createStagingBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage) {
    return createBuffer(
        size, usage,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
  }
  Image VulkanContext::create2DImage(const vk::Extent2D& dim, vk::Format format,
                                     vk::ImageUsageFlags usage, vk::MemoryPropertyFlags props,
                                     bool mipmaps, bool createView) {
    Image image{*this};
    auto img = device.createImage(vk::ImageCreateInfo{}
                                      .setImageType(vk::ImageType::e2D)
                                      .setFormat(format)
                                      .setExtent({dim.width, dim.height, (u32)1})
                                      .setMipLevels((mipmaps ? get_num_mip_levels(dim) : 1))
                                      .setArrayLayers(1)
                                      .setUsage(usage | vk::ImageUsageFlagBits::eTransferSrc
                                                | vk::ImageUsageFlagBits::eTransferDst)
                                      .setSamples(vk::SampleCountFlagBits::e1),
                                  nullptr, dispatcher);

    vk::MemoryRequirements memRequirements = device.getImageMemoryRequirements(img, dispatcher);
    u32 memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, props);
    vk::MemoryAllocateInfo allocInfo{memRequirements.size, memoryTypeIndex};
    auto mem = device.allocateMemory(allocInfo, nullptr, dispatcher);

    device.bindImageMemory(img, mem, 0, dispatcher);

    VkMemory memory{*this};
    memory.mem = mem;
    memory.memSize = memRequirements.size;
    memory.memoryPropertyFlags = memoryProperties.memoryTypes[memoryTypeIndex].propertyFlags;

    image.image = img;
    image.pmem = std::make_shared<VkMemory>(std::move(memory));
    if (createView) {
      image.pview = device.createImageView(
          vk::ImageViewCreateInfo{}
              .setImage(img)
              .setPNext(nullptr)
              .setViewType(vk::ImageViewType::e2D)
              .setFormat(format)
              .setSubresourceRange(vk::ImageSubresourceRange{
                  is_depth_format(format) ? vk::ImageAspectFlagBits::eDepth
                                          : vk::ImageAspectFlagBits::eColor,
                  0, 1 /*VK_REMAINING_MIP_LEVELS*/, 0, 1
                  /*VK_REMAINING_ARRAY_LAYERS*/}),
          nullptr, dispatcher);
    }
    return image;
  }
  ImageView VulkanContext::create2DImageView(vk::Image image, vk::Format format,
                                             vk::ImageAspectFlags aspect, u32 levels,
                                             const void* pNextImageView) {
    ImageView imgv{*this};
    imgv.imgv = device.createImageView(
        vk::ImageViewCreateInfo{}
            .setImage(image)
            .setPNext(pNextImageView)
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(format)
            .setSubresourceRange(vk::ImageSubresourceRange{aspect, 0, levels, 0, 1}),
        nullptr, dispatcher);
    return imgv;
  }
  Framebuffer VulkanContext::createFramebuffer(const std::vector<vk::ImageView>& imageViews,
                                               vk::Extent2D extent, vk::RenderPass renderPass) {
    Framebuffer obj{*this};
    auto ci = vk::FramebufferCreateInfo{
        {},    renderPass, (u32)imageViews.size(), imageViews.data(), extent.width, extent.height,
        (u32)1};
    obj.framebuffer = device.createFramebuffer(ci, nullptr, dispatcher);
    return obj;
  }

  DescriptorPool VulkanContext::createDescriptorPool(
      const std::vector<vk::DescriptorPoolSize>& poolSizes, u32 maxSets) {
    /// @note DescriptorPoolSize: descriptorCount, vk::DescriptorType::eUniformBufferDynamic
    auto poolCreateInfo = vk::DescriptorPoolCreateInfo()
                              .setMaxSets(maxSets)
                              .setPoolSizeCount((u32)poolSizes.size())
                              .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
                              .setPPoolSizes(poolSizes.data());
    DescriptorPool ret{*this};
    ret.descriptorPool = device.createDescriptorPool(poolCreateInfo, nullptr, dispatcher);
    return ret;
  }
  ShaderModule VulkanContext::createShaderModule(const std::vector<char>& code) {
    if (code.size() & (sizeof(u32) - 1))
      throw std::runtime_error(
          "the number of bytes of the spirv code should be a multiple of u32 type size.");
    return createShaderModule(reinterpret_cast<const u32*>(code.data()), code.size() / sizeof(u32));
  }
  ShaderModule VulkanContext::createShaderModule(const u32* code, size_t size) {
    ShaderModule ret{*this};
    vk::ShaderModuleCreateInfo smCI{{}, size * sizeof(u32), code};
    ret.shaderModule = device.createShaderModule(smCI, nullptr, dispatcher);
    ret.analyzeLayout(code, size);
    return ret;
  }

  ///
  ///
  /// working context (CmdContext)
  ///
  ///
  ExecutionContext::ExecutionContext(VulkanContext& ctx)
      : ctx{ctx}, poolFamilies(ctx.numDistinctQueueFamilies()) {
    for (const auto& [family, queueFamilyIndex] : zip(poolFamilies, ctx.uniqueQueueFamilyIndices)) {
      family.reusePool = ctx.device.createCommandPool(
          vk::CommandPoolCreateInfo{{}, queueFamilyIndex}, nullptr, ctx.dispatcher);
      /// @note for memory allcations, etc.
      family.singleUsePool = ctx.device.createCommandPool(
          vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eTransient, queueFamilyIndex},
          nullptr, ctx.dispatcher);
      family.resetPool = ctx.device.createCommandPool(
          vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eTransient
                                        | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                    queueFamilyIndex},
          nullptr, ctx.dispatcher);
      family.queue = ctx.device.getQueue(queueFamilyIndex, 0, ctx.dispatcher);
      family.pctx = &ctx;
    }
  }
  ExecutionContext::~ExecutionContext() {
    for (auto& family : poolFamilies) {
#if 0
      // reset and reuse
      for (auto& cmd : family.cmds)
        ctx.device.freeCommandBuffers(family.singleUsePool, cmd, ctx.dispatcher);
      family.cmds.clear();
#endif

      ctx.device.resetCommandPool(family.reusePool, vk::CommandPoolResetFlagBits::eReleaseResources,
                                  ctx.dispatcher);
      ctx.device.destroyCommandPool(family.reusePool, nullptr, ctx.dispatcher);

      ctx.device.resetCommandPool(family.singleUsePool,
                                  vk::CommandPoolResetFlagBits::eReleaseResources, ctx.dispatcher);
      ctx.device.destroyCommandPool(family.singleUsePool, nullptr, ctx.dispatcher);

      ctx.device.resetCommandPool(family.resetPool, vk::CommandPoolResetFlagBits::eReleaseResources,
                                  ctx.dispatcher);
      ctx.device.destroyCommandPool(family.resetPool, nullptr, ctx.dispatcher);
    }
  }

}  // namespace zs