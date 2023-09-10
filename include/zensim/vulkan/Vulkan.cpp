#include "Vulkan.hpp"

#include <vulkan/vulkan_structs.hpp>

#include "zensim/Logger.hpp"
#include "zensim/Platform.hpp"
#include "zensim/ZpcReflection.hpp"
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

  /// @ref:
  /// https://github.com/KhronosGroup/Vulkan-Hpp/blob/main/README.md#extensions--per-device-function-pointers
  Vulkan::Vulkan() {
    /// @note instance
    vk::DynamicLoader dl;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr
        = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    _dispatcher.init(vkGetInstanceProcAddr);

    vk::ApplicationInfo appInfo{"zpc_app", 0, "zpc", 0, VK_API_VERSION_1_3};

    std::vector<const char*> extensions = {"VK_KHR_surface", "VK_EXT_debug_utils"};
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
    for (int i = 0; i != physicalDevices.size(); ++i) {
      auto& physDev = physicalDevices[i];
      _contexts.emplace_back(i, physDev, _dispatcher);
    }
  }
  Vulkan::~Vulkan() {
    /// @note clear contexts
    for (auto& ctx : _contexts) {
      ctx.device.destroy(nullptr, ctx.dispatcher);
    }
    _contexts.clear();

    /// @note clear instance-created objects
    if (_messenger) _instance.destroy(_messenger, nullptr, _dispatcher);

    /// @note destroy instance itself
    _instance.destroy(nullptr, _dispatcher);
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
    u32 graphicsQueueFamilyIndex = queueFamilyProps.size();
    for (int i = 0; i != queueFamilyProps.size(); ++i) {
      auto& q = queueFamilyProps[i];
      if (q.queueFlags & vk::QueueFlagBits::eGraphics) {
        graphicsQueueFamilyIndex = i;
        ZS_WARN_IF(!(q.queueFlags & vk::QueueFlagBits::eTransfer),
                   "the selected graphics queue family cannot transfer!");
        break;
      }
    }
    ZS_ERROR_IF(graphicsQueueFamilyIndex == queueFamilyProps.size(), "graphics");
    fmt::print("selected queue family [{}] for graphics!\n", graphicsQueueFamilyIndex);

    float priority = 1.f;
    vk::DeviceQueueCreateInfo dqCI{{}, graphicsQueueFamilyIndex, 1, &priority};

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
    vk::DeviceCreateInfo devCI{
        {}, 1, &dqCI, 0, nullptr, (u32)enabledExtensions.size(), enabledExtensions.data()};

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
        "\n\t\t(Graphics) queue family index: {}."
        "\n\tEnabled the following device tensions ({} in total):\n",
        devid, devProps.deviceName, graphicsQueueFamilyIndex, enabledExtensions.size());
    for (auto ext : enabledExtensions) fmt::print("\t\t{}\n", ext);
  }

}  // namespace zs
