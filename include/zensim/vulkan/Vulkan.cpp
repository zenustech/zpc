#include "Vulkan.hpp"

#include <iostream>
#include <map>
#include <set>
#include <thread>

#include "zensim/Logger.hpp"
#include "zensim/Platform.hpp"
#include "zensim/ZpcReflection.hpp"
#include "zensim/types/Iterator.h"
#include "zensim/types/SourceLocation.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

#define MEM_POOL_CTRL 3

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

    fmt::print(
        "\t[InitInfo -- Dev Property] Vulkan device [{}] name: {}."
        "\n\t\t(Graphics/Compute/Transfer) queue family index: {}, {}, {}. Ray-tracing support: "
        "{}. "
        "\n\tEnabled the following device tensions ({} in total):\n",
        devid, devProps.deviceName, graphicsQueueFamilyIndex, computeQueueFamilyIndex,
        transferQueueFamilyIndex, rtPreds == rtRequiredPreds, enabledExtensions.size());
    for (auto ext : enabledExtensions) fmt::print("\t\t{}\n", ext);
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

  ///
  ///
  /// swapchain
  ///
  ///
  SwapchainBuilder::SwapchainBuilder(Vulkan::VulkanContext& ctx, vk::SurfaceKHR targetSurface)
      : ctx{ctx}, surface{targetSurface} {
    ZS_ERROR_IF(
        !ctx.supportSurface(surface),
        fmt::format("queue [{}] does not support this surface!\n", ctx.graphicsQueueFamilyIndex));
    surfFormats = ctx.physicalDevice.getSurfaceFormatsKHR(surface, ctx.dispatcher);
    surfCapabilities = ctx.physicalDevice.getSurfaceCapabilitiesKHR(surface, ctx.dispatcher);
    surfPresentModes = ctx.physicalDevice.getSurfacePresentModesKHR(surface, ctx.dispatcher);

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
  void SwapchainBuilder::presentMode(vk::PresentModeKHR mode) {
    if (mode == ci.presentMode) return;
    for (auto& m : surfPresentModes)
      if (m == mode) {
        ci.presentMode = mode;
        return;
      }
    ZS_WARN(fmt::format("Present mode [{}] is not supported in this context. Ignored.\n", mode));
  }
  void SwapchainBuilder::build(Swapchain& obj) {
    // kept the previously built swapchain for this
    ci.oldSwapchain = obj.swapchain;
    obj.swapchain = ctx.device.createSwapchainKHR(ci, nullptr, ctx.dispatcher);
    obj.frameIndex = 0;
    obj.images = ctx.device.getSwapchainImagesKHR(obj.swapchain, ctx.dispatcher);

    /// reset previous resources (if any)
    obj.resetAux();
    if (ci.oldSwapchain) ctx.device.destroySwapchainKHR(ci.oldSwapchain, nullptr, ctx.dispatcher);

    /// construct current swapchain
    obj.imageViews.resize(obj.images.size());
    obj.readSemaphores.resize(obj.images.size());
    obj.writeSemaphores.resize(obj.images.size());
    obj.readFences.resize(obj.images.size());
    obj.writeFences.resize(obj.images.size());
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
      // semaphores
      obj.readSemaphores[i]
          = ctx.device.createSemaphore(vk::SemaphoreCreateInfo{}, nullptr, ctx.dispatcher);
      obj.writeSemaphores[i]
          = ctx.device.createSemaphore(vk::SemaphoreCreateInfo{}, nullptr, ctx.dispatcher);
      obj.readFences[i] = ctx.device.createFence(vk::FenceCreateInfo{}, nullptr, ctx.dispatcher);
      obj.writeFences[i] = ctx.device.createFence(vk::FenceCreateInfo{}, nullptr, ctx.dispatcher);
    }
  }

}  // namespace zs
