#include "Vulkan.hpp"

#include <iostream>
#include <map>
#include <set>
#include <thread>
// resources
#include "zensim/vulkan/VkBuffer.hpp"
#include "zensim/vulkan/VkDescriptor.hpp"
#include "zensim/vulkan/VkImage.hpp"
#include "zensim/vulkan/VkPipeline.hpp"
#include "zensim/vulkan/VkRenderPass.hpp"

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

  /// @ref: dokipen3d/vulkanHppMinimalExample
  static VKAPI_ATTR VkBool32 VKAPI_CALL
  zsvk_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                      VkDebugUtilsMessageTypeFlagsEXT messageType,
                      const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
  }
  using ContextEnvs = std::map<int, ExecutionContext>;
  using WorkerEnvs = std::map<std::thread::id, ContextEnvs>;
  namespace {
    static WorkerEnvs g_workingContexts;
    static Mutex g_mtx{};
  }  // namespace

  /// @ref:
  /// https://github.com/KhronosGroup/Vulkan-Hpp/blob/main/README.md#extensions--per-device-function-pointers
  Vulkan::Vulkan() {
    /// @note instance
    vk::DynamicLoader dl;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr
        = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    _dispatcher.init(vkGetInstanceProcAddr);

    vk::ApplicationInfo appInfo{"zpc_app", 0, "zpc", 0, VK_API_VERSION_1_3};

    /// @ref: VkBootstrap
    std::vector<const char*> extensions
        = { "VK_KHR_surface",
            "VK_EXT_debug_utils",
#if defined(ZS_PLATFORM_WINDOWS)
            "VK_KHR_win32_surface"
#elif defined(ZS_PLATFORM_OSX)
            "VK_EXT_metal_surface"
#elif defined(ZS_PLATFORM_LINUX)
            "VK_KHR_xcb_surface"  // or "VK_KHR_xlib_surface", "VK_KHR_wayland_surface"
#else
            static_assert(false, "unsupported platform for vulkan instance creation!");
#endif
          };
    std::vector<const char*> enabledLayers = {"VK_LAYER_KHRONOS_validation"};
    vk::InstanceCreateInfo instCI{{},
                                  &appInfo,
                                  (u32)enabledLayers.size(),
                                  enabledLayers.data(),
                                  (u32)extensions.size(),
                                  extensions.data()};

    _instance = vk::createInstance(instCI);

#if 0
    _dispatcher = vk::DispatchLoaderDynamic(_instance, vkGetInstanceProcAddr);
#else
    _dispatcher.init(_instance);
#endif
    _messenger = _instance.createDebugUtilsMessengerEXT(
        vk::DebugUtilsMessengerCreateInfoEXT{
            {},
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
                | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
            // | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
            // | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo
            ,
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
                | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
                | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            zsvk_debug_callback},
        nullptr, _dispatcher);

    /// @note physical devices and their contexts
    auto physicalDevices = _instance.enumeratePhysicalDevices(_dispatcher);
    fmt::print("\t[InitInfo -- DevNum] Detected {} Vulkan Capable physical device(s)\n",
               physicalDevices.size());

    _defaultContext = -1;
    for (int i = 0; i != physicalDevices.size(); ++i) {
      auto& physDev = physicalDevices[i];
      _contexts.emplace_back(i, physDev, _dispatcher);
      if (_defaultContext == -1 && _contexts.back().supportGraphics()) _defaultContext = i;
    }
  }  // namespace zs
  Vulkan::~Vulkan() {
    /// @note clear contexts
    for (auto& ctx : _contexts) ctx.reset();
    _contexts.clear();

    /// @note clear instance-created objects
    if (_messenger) _instance.destroy(_messenger, nullptr, _dispatcher);

    /// @note destroy instance itself
    _instance.destroy(nullptr, _dispatcher);
    _instance = vk::Instance{};
    fmt::print("zpc vulkan instance has been destroyed.\n");
  }

  ///
  ///
  /// vulkan context
  ///
  ///
  void Vulkan::VulkanContext::reset() {
    /// clear builders
    // if (swapchainBuilder) swapchainBuilder.reset(nullptr);
    /// clear resources
    {
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
    /// destroy logical device
    device.destroy(nullptr, dispatcher);
    fmt::print("vulkan context [{}] (of {}) has been successfully reset.\n", devid,
               driver().num_devices());
  }

  Vulkan::VulkanContext::VulkanContext(int devId, vk::PhysicalDevice phydev,
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

  ExecutionContext& Vulkan::VulkanContext::env() {
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

  Buffer Vulkan::VulkanContext::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
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
  Buffer Vulkan::VulkanContext::createStagingBuffer(vk::DeviceSize size,
                                                    vk::BufferUsageFlags usage) {
    return createBuffer(
        size, usage,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
  }
  Image Vulkan::VulkanContext::create2DImage(const vk::Extent2D& dim, vk::Format format,
                                             vk::ImageUsageFlags usage,
                                             vk::MemoryPropertyFlags props, bool mipmaps,
                                             bool createView) {
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
  ImageView Vulkan::VulkanContext::create2DImageView(vk::Image image, vk::Format format,
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
  Framebuffer Vulkan::VulkanContext::createFramebuffer(const std::vector<vk::ImageView>& imageViews,
                                                       vk::Extent2D extent,
                                                       vk::RenderPass renderPass) {
    Framebuffer obj{*this};
    auto ci = vk::FramebufferCreateInfo{
        {},    renderPass, (u32)imageViews.size(), imageViews.data(), extent.width, extent.height,
        (u32)1};
    obj.framebuffer = device.createFramebuffer(ci, nullptr, dispatcher);
    return obj;
  }

  DescriptorPool Vulkan::VulkanContext::createDescriptorPool(
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

  ///
  ///
  /// working context (CmdContext)
  ///
  ///
  ExecutionContext::ExecutionContext(Vulkan::VulkanContext& ctx)
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

  ///
  ///
  /// swapchain builder
  ///
  ///
  u32 Swapchain::acquireNextImage() {
    if (vk::Result res = ctx.device.waitForFences(
            1, &readFences[frameIndex], VK_TRUE, detail::deduce_numeric_max<u64>(), ctx.dispatcher);
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
  RenderPass Swapchain::getRenderPass() {
    RenderPass ret{ctx};
    const bool includeDepthBuffer = depthBuffers.size() == num_buffered_frames;
    std::vector<vk::AttachmentDescription> attachments(1 + (includeDepthBuffer ? 1 : 0));
    std::vector<vk::AttachmentReference> refs(attachments.size());
    // color
    auto& colorAttachment = attachments[0];
    colorAttachment = vk::AttachmentDescription{}
                          .setFormat(colorFormat)
                          .setSamples(vk::SampleCountFlagBits::e1)
                          .setLoadOp(vk::AttachmentLoadOp::eClear)
                          .setStoreOp(vk::AttachmentStoreOp::eStore)
                          .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                          .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                          .setInitialLayout(vk::ImageLayout::eUndefined)
                          .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

    auto& colorAttachmentRef = refs[0];
    colorAttachmentRef = vk::AttachmentReference{0, vk::ImageLayout::eColorAttachmentOptimal};

    // depth
    if (includeDepthBuffer) {
      auto& depthAttachment = attachments[1];
      depthAttachment = vk::AttachmentDescription{}
                            .setFormat(depthFormat)
                            .setSamples(vk::SampleCountFlagBits::e1)
                            .setLoadOp(vk::AttachmentLoadOp::eClear)
                            .setStoreOp(vk::AttachmentStoreOp::eDontCare)
                            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                            .setInitialLayout(vk::ImageLayout::eUndefined)
                            .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

      auto& depthAttachmentRef = refs[1];
      depthAttachmentRef
          = vk::AttachmentReference{1, vk::ImageLayout::eDepthStencilAttachmentOptimal};
    }

    /// @note here color attachment index corresponding to
    /// 'layout(location = k) out vec4 outColor' directive in the fragment shader
    auto subpass = vk::SubpassDescription{}
                       .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                       .setColorAttachmentCount(1)
                       .setPColorAttachments(&refs[0]);
    if (includeDepthBuffer) subpass.setPDepthStencilAttachment(&refs[1]);

    vk::AccessFlags accessFlag;
    if (includeDepthBuffer)
      accessFlag = vk::AccessFlagBits::eColorAttachmentWrite
                   | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    else
      accessFlag = vk::AccessFlagBits::eColorAttachmentWrite;
    auto dependency = vk::SubpassDependency{}
                          .setDstSubpass(0)
                          .setDstAccessMask(accessFlag)
                          .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput
                                           | vk::PipelineStageFlagBits::eEarlyFragmentTests)
                          .setSrcSubpass(VK_SUBPASS_EXTERNAL)
                          .setSrcAccessMask({})
                          .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput
                                           | vk::PipelineStageFlagBits::eEarlyFragmentTests);

    ret.renderpass = ctx.device.createRenderPass(
        vk::RenderPassCreateInfo{
            {}, (u32)attachments.size(), attachments.data(), (u32)1, &subpass, (u32)1, &dependency},
        nullptr, ctx.dispatcher);
    return ret;
  }
  void Swapchain::initFramebuffersFor(vk::RenderPass renderPass) {
    frameBuffers.clear();
    auto cnt = imageCount();
    if (depthBuffers.size() != cnt) {
      // color + depth
      for (int i = 0; i != cnt; ++i) {
        frameBuffers.emplace_back(ctx.createFramebuffer(
            {(vk::ImageView)imageViews[i], (vk::ImageView)depthBuffers[i]}, extent, renderPass));
      }
    } else {
      // color only
      for (int i = 0; i != imageCount(); ++i) {
        frameBuffers.emplace_back(
            ctx.createFramebuffer({(vk::ImageView)imageViews[i]}, extent, renderPass));
      }
    }
  }

  SwapchainBuilder::SwapchainBuilder(Vulkan::VulkanContext& ctx, vk::SurfaceKHR targetSurface)
      : ctx{ctx}, surface{targetSurface} {
    ZS_ERROR_IF(
        !ctx.supportSurface(surface),
        fmt::format("queue [{}] does not support this surface!\n", ctx.graphicsQueueFamilyIndex));
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
      ci.imageFormat = vk::Format::eR8G8B8A8Srgb;
      ci.imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
    } else {
      ci.imageFormat = surfFormats.front().format;
      ci.imageColorSpace = surfFormats.front().colorSpace;
      for (const auto& fmt : surfFormats)
        if (fmt.format == vk::Format::eR8G8B8A8Srgb
            && fmt.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
          ci.imageFormat = vk::Format::eR8G8B8A8Srgb;
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
    constexpr auto num_buffered_frames = Swapchain::num_buffered_frames;
    // kept the previously built swapchain for this
    ci.oldSwapchain = obj.swapchain;
    obj.swapchain = ctx.device.createSwapchainKHR(ci, nullptr, ctx.dispatcher);
    obj.frameIndex = 0;
    obj.extent = ci.imageExtent;
    obj.colorFormat = ci.imageFormat;
    obj.depthFormat = swapchainDepthFormat;
    obj.images = ctx.device.getSwapchainImagesKHR(obj.swapchain, ctx.dispatcher);

    /// reset previous resources (if any)
    obj.resetAux();
    if (ci.oldSwapchain) ctx.device.destroySwapchainKHR(ci.oldSwapchain, nullptr, ctx.dispatcher);

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
    if (obj.readSemaphores.size() != num_buffered_frames) {
      obj.readSemaphores.resize(num_buffered_frames);
      obj.writeSemaphores.resize(num_buffered_frames);
      obj.readFences.resize(num_buffered_frames);
      obj.writeFences.resize(num_buffered_frames);
      for (int i = 0; i != num_buffered_frames; ++i) {
        // semaphores
        obj.readSemaphores[i]
            = ctx.device.createSemaphore(vk::SemaphoreCreateInfo{}, nullptr, ctx.dispatcher);
        obj.writeSemaphores[i]
            = ctx.device.createSemaphore(vk::SemaphoreCreateInfo{}, nullptr, ctx.dispatcher);
        obj.readFences[i] = ctx.device.createFence(vk::FenceCreateInfo{}, nullptr, ctx.dispatcher);
        obj.writeFences[i] = ctx.device.createFence(vk::FenceCreateInfo{}, nullptr, ctx.dispatcher);
      }
    }
  }

}  // namespace zs
