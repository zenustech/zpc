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
#include "zensim/vulkan/VkSwapchain.hpp"

//
#include "zensim/Logger.hpp"
#include "zensim/Platform.hpp"
#include "zensim/ZpcReflection.hpp"
#include "zensim/execution/ConcurrencyPrimitive.hpp"
#include "zensim/types/Iterator.h"
#include "zensim/types/SourceLocation.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

namespace {
  std::set<const char*> g_vulkanInstanceExtensions;
  std::map<int, std::set<const char*>> g_vulkanDeviceExtensions;
}  // namespace

namespace zs {

  using ContextEnvs = std::map<int, ExecutionContext>;
  using WorkerEnvs = std::map<std::thread::id, ContextEnvs>;

  /// @ref: dokipen3d/vulkanHppMinimalExample
  static VKAPI_ATTR VkBool32 VKAPI_CALL
  zsvk_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                      VkDebugUtilsMessageTypeFlagsEXT messageType,
                      const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
  }

  Vulkan& Vulkan::instance() {
    static Vulkan s_instance{};
    return s_instance;
  }

  Vulkan &Vulkan::driver() noexcept { return instance(); }
  size_t Vulkan::num_devices() noexcept { return instance()._contexts.size(); }
  vk::Instance Vulkan::vk_inst() noexcept { return instance()._instance; }
  const vk::DispatchLoaderDynamic &Vulkan::vk_inst_dispatcher() noexcept {
      return instance()._dispatcher;
  }
  VulkanContext &Vulkan::context(int devid) { return driver()._contexts[devid]; }
  VulkanContext &Vulkan::context() { return instance()._contexts[instance()._defaultContext]; }



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
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
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
#ifdef ZS_PLATFORM_OSX
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif
    std::vector<const char*> enabledLayers = {
#if ZS_ENABLE_VULKAN_VALIDATION
      "VK_LAYER_KHRONOS_validation"
#endif
    };
    vk::InstanceCreateInfo instCI{{},
                                  &appInfo,
                                  (u32)enabledLayers.size(),
                                  enabledLayers.data(),
                                  (u32)extensions.size(),
                                  extensions.data()};
#ifdef ZS_PLATFORM_OSX
    instCI.flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

    std::vector enabledValidationFeatures{vk::ValidationFeatureEnableEXT::eDebugPrintf};
    vk::ValidationFeaturesEXT validationFeatures;
    validationFeatures.enabledValidationFeatureCount = (u32)enabledValidationFeatures.size();
    validationFeatures.pEnabledValidationFeatures = enabledValidationFeatures.data();
    instCI.setPNext(&validationFeatures);

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
      _contexts.emplace_back(i, _instance, physDev, _dispatcher);
      if (_defaultContext == -1 && _contexts.back().supportGraphics()) _defaultContext = i;
    }

    _workingContexts = new WorkerEnvs();
    _mutex = new Mutex();
  }  // namespace zs
  void Vulkan::reset() {
#if 0
    {
      WorkerEnvs &g_workingContexts = working_contexts<WorkerEnvs>();
      Mutex &g_mtx = working_mutex<Mutex>();
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
#endif
    if (_workingContexts) {
      delete static_cast<WorkerEnvs *>(_workingContexts);
      _workingContexts = nullptr;
    }
    if (_mutex) {
      delete static_cast<Mutex *>(_mutex);
      _mutex = nullptr;
    }
    /// @note clear contexts
    for (auto& ctx : _contexts) ctx.reset();
    _contexts.clear();

    /// @note clear instance-created objects
    if (_instance) {
      /// @note may destroy window surface, etc.
      if (_onDestroyCallback) _onDestroyCallback();

      _instance.destroy(_messenger, nullptr, _dispatcher);

      /// @note destroy instance itself
      _instance.destroy(nullptr, _dispatcher);
      _instance = vk::Instance{};
    }
  }
  Vulkan::~Vulkan() {
    reset();
    fmt::print("zpc vulkan instance has been destroyed.\n");
  }

}  // namespace zs
