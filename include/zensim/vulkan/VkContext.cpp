// vulkan memory allocator impl
#define VK_ENABLE_BETA_EXTENSIONS
// to use VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR
#include "vulkan/vulkan_core.h"
#include "zensim/Platform.hpp"
//
#if defined(ZS_PLATFORM_OSX)
#  include "vulkan/vulkan_beta.h"
#endif
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include "vma/vk_mem_alloc.h"
//
#include "zensim/vulkan/VkContext.hpp"
//
#include <iostream>
#include <map>
#include <set>
#include <thread>

#include "zensim/vulkan/VkBuffer.hpp"
#include "zensim/vulkan/VkCommand.hpp"
#include "zensim/vulkan/VkDescriptor.hpp"
#include "zensim/vulkan/VkImage.hpp"
#include "zensim/vulkan/VkPipeline.hpp"
#include "zensim/vulkan/VkRenderPass.hpp"
#include "zensim/vulkan/VkShader.hpp"
#include "zensim/vulkan/VkSwapchain.hpp"
#include "zensim/vulkan/Vulkan.hpp"

//
#include "zensim/Logger.hpp"
#include "zensim/ZpcReflection.hpp"
#include "zensim/execution/ConcurrencyPrimitive.hpp"
#include "zensim/types/Iterator.h"
#include "zensim/types/SourceLocation.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"
#include "zensim/zpc_tpls/magic_enum/magic_enum.hpp"

namespace zs {

  using ContextEnvs = std::map<int, ExecutionContext>;
  using WorkerEnvs = std::map<std::thread::id, ContextEnvs>;

  // now moved to Vulkan singleton
  // namespace {
  //   static WorkerEnvs g_workingContexts;
  //   static Mutex g_mtx{};
  // }  // namespace

  ///
  ///
  /// vulkan swapchain builder
  ///
  ///
  SwapchainBuilderOwner::SwapchainBuilderOwner(void* handle) noexcept : _handle{handle} {}
  SwapchainBuilderOwner::~SwapchainBuilderOwner() {
    if (_handle) {
      delete static_cast<SwapchainBuilder*>(_handle);
      _handle = nullptr;
    }
  }

  SwapchainBuilderOwner::SwapchainBuilderOwner(SwapchainBuilderOwner&& o) noexcept
      : _handle{zs::exchange(o._handle, nullptr)} {}

  SwapchainBuilderOwner& SwapchainBuilderOwner::operator=(SwapchainBuilderOwner&& o) {
    if (_handle) delete static_cast<SwapchainBuilder*>(_handle);
    _handle = zs::exchange(o._handle, nullptr);
    return *this;
  }

  void SwapchainBuilderOwner::reset(void* handle) {
    if (_handle) delete static_cast<SwapchainBuilder*>(_handle);
    _handle = handle;
  }

  ///
  ///
  /// vulkan context
  ///
  ///
  Vulkan& VulkanContext::driver() const noexcept { return Vulkan::driver(); }

  void VulkanContext::reset() {
    /// clear builders
    // if (swapchainBuilder) swapchainBuilder.reset(nullptr);
    /// clear execution resources
    // handled by Vulkan

    destructDescriptorPool();

    vmaDestroyAllocator(defaultAllocator);
    defaultAllocator = 0;  // ref: nvpro-core

    /// destroy logical device
    if (device) {
      device.destroy(nullptr, dispatcher);
      device = vk::Device{};
    }
    fmt::print("vulkan context [{}] (of {}) has been successfully reset.\n", devid,
               driver().num_devices());
  }

  VulkanContext::VulkanContext(int devId, vk::Instance instance, vk::PhysicalDevice phydev,
                               const vk::DispatchLoaderDynamic& instDispatcher)
      : devid{devId}, physicalDevice{phydev}, device{}, dispatcher{instDispatcher} {
    /// @note logical device
    std::vector<vk::ExtensionProperties> devExts
        = physicalDevice.enumerateDeviceExtensionProperties();
    vk::PhysicalDeviceProperties devProps = physicalDevice.getProperties();

    /// queue family
    queueFamilyProps = physicalDevice.getQueueFamilyProperties();
    for (auto& queueFamilyIndex : queueFamilyIndices) queueFamilyIndex = -1;
    for (auto& queueFamilyMap : queueFamilyMaps) queueFamilyMap = -1;
    int graphicsAndCompute = -1;
    for (int i = 0; i != queueFamilyProps.size(); ++i) {
      int both = 0;
      auto& q = queueFamilyProps[i];
      if (q.queueCount == 0) continue;
      if (queueFamilyIndices[vk_queue_e::graphics] == -1
          && (q.queueFlags & vk::QueueFlagBits::eGraphics)) {
        queueFamilyIndices[vk_queue_e::graphics] = i;
        ZS_WARN_IF(!(q.queueFlags & vk::QueueFlagBits::eTransfer),
                   "the selected graphics queue family cannot transfer!");
        both++;
      }
      if (queueFamilyIndices[vk_queue_e::compute] == -1
          && (q.queueFlags & vk::QueueFlagBits::eCompute)) {
        queueFamilyIndices[vk_queue_e::compute] = i;
        both++;
      }
      if (queueFamilyIndices[vk_queue_e::transfer] == -1
          && (q.queueFlags & vk::QueueFlagBits::eTransfer))
        queueFamilyIndices[vk_queue_e::transfer] = i;

      if (both == 2) graphicsAndCompute = i;

      if (queueFamilyIndices[vk_queue_e::dedicated_transfer] == -1
          && (q.queueFlags & vk::QueueFlagBits::eTransfer) && both == 0) {
        queueFamilyIndices[vk_queue_e::dedicated_transfer] = i;
      }

      /// queue family info
      fmt::print(
          "\n\t====> {}-th queue family has {} queue(s).\n\tQueue "
          "capabilities [graphics: {}, compute: {}, transfer: {}, sparse binding: {}, \n\t\tvideo "
          "encode: {}, video decode: {}]\n",
          i, q.queueCount, static_cast<bool>(q.queueFlags & vk::QueueFlagBits::eGraphics),
          static_cast<bool>(q.queueFlags & vk::QueueFlagBits::eCompute),
          static_cast<bool>(q.queueFlags & vk::QueueFlagBits::eTransfer),
          static_cast<bool>(q.queueFlags & vk::QueueFlagBits::eSparseBinding),
          static_cast<bool>(q.queueFlags & vk::QueueFlagBits::eVideoEncodeKHR),
          static_cast<bool>(q.queueFlags & vk::QueueFlagBits::eVideoDecodeKHR));
    }
    if (graphicsAndCompute == -1)
      throw std::runtime_error(
          "there should be at least a queue that supports both graphics and compute!");
    queueFamilyIndices[vk_queue_e::graphics] = queueFamilyIndices[vk_queue_e::compute]
        = graphicsAndCompute;

    for (int i = 0; i != queueFamilyProps.size(); ++i) {
      auto& q = queueFamilyProps[i];
      if (q.queueCount == 0 || !(q.queueFlags & vk::QueueFlagBits::eCompute)
          || i == queueFamilyIndices[vk_queue_e::graphics])
        continue;
      if (queueFamilyIndices[vk_queue_e::dedicated_compute] == -1
          || i != queueFamilyIndices[vk_queue_e::dedicated_transfer])
        queueFamilyIndices[vk_queue_e::dedicated_compute] = i;
    }

    ZS_ERROR_IF(queueFamilyIndices[vk_queue_e::graphics] == -1,
                "graphics queue family does not exist!");
    fmt::print(
        "selected queue family [{}] for graphics! (compute: {}, transfer: {}, dedicated compute: "
        "{}, dedicated transfer: {})\n",
        queueFamilyIndices[vk_queue_e::graphics], queueFamilyIndices[vk_queue_e::compute],
        queueFamilyIndices[vk_queue_e::transfer], queueFamilyIndices[vk_queue_e::dedicated_compute],
        queueFamilyIndices[vk_queue_e::dedicated_transfer]);

    std::set<u32> uniqueQueueFamilyIndices{(u32)queueFamilyIndices[vk_queue_e::graphics],
                                           (u32)queueFamilyIndices[vk_queue_e::compute],
                                           (u32)queueFamilyIndices[vk_queue_e::transfer],
                                           (u32)queueFamilyIndices[vk_queue_e::dedicated_compute],
                                           (u32)queueFamilyIndices[vk_queue_e::dedicated_transfer]};
    uniqueQueueFamilyIndices.erase(-1);
    this->uniqueQueueFamilyIndices.reserve(uniqueQueueFamilyIndices.size());
    std::vector<vk::DeviceQueueCreateInfo> dqCIs(uniqueQueueFamilyIndices.size());
    std::vector<std::vector<float>> uniqueQueuePriorities(uniqueQueueFamilyIndices.size());
    {
      u32 i = 0;
      for (auto index : uniqueQueueFamilyIndices) {
        const auto& queueFamilyProp = queueFamilyProps[i];
        this->uniqueQueueFamilyIndices.push_back(index);
        uniqueQueuePriorities[i].resize(queueFamilyProp.queueCount);
        for (auto& v : uniqueQueuePriorities[i]) v = 0.5f;
        dqCIs[i]
            .setQueueCount(queueFamilyProp.queueCount)
            .setQueueFamilyIndex(index)
            .setQueuePriorities(uniqueQueuePriorities[i]);
        // .setPQueuePriorities(uniqueQueuePriorities[i].data());

        if (queueFamilyIndices[vk_queue_e::graphics] == index)
          queueFamilyMaps[vk_queue_e::graphics] = i;
        if (queueFamilyIndices[vk_queue_e::compute] == index)
          queueFamilyMaps[vk_queue_e::compute] = i;
        if (queueFamilyIndices[vk_queue_e::transfer] == index)
          queueFamilyMaps[vk_queue_e::transfer] = i;
        if (queueFamilyIndices[vk_queue_e::dedicated_compute] == index)
          queueFamilyMaps[vk_queue_e::dedicated_compute] = i;
        if (queueFamilyIndices[vk_queue_e::dedicated_transfer] == index)
          queueFamilyMaps[vk_queue_e::dedicated_transfer] = i;

        i++;
      }
      fmt::print(
          "queue family maps [graphics: {} ({} queues), compute: {} ({} queues), transfer: {} ({} "
          "queues), dedicated compute: {} ({} queues), dedicated transfer: {} ({} queues)]\n",
          queueFamilyMaps[vk_queue_e::graphics],
          queueFamilyProps[queueFamilyMaps[vk_queue_e::graphics]].queueCount,
          queueFamilyMaps[vk_queue_e::compute],
          queueFamilyProps[queueFamilyMaps[vk_queue_e::compute]].queueCount,
          queueFamilyMaps[vk_queue_e::transfer],
          queueFamilyProps[queueFamilyMaps[vk_queue_e::transfer]].queueCount,
          queueFamilyMaps[vk_queue_e::dedicated_compute] != -1
              ? queueFamilyMaps[vk_queue_e::dedicated_compute]
              : -1,
          queueFamilyMaps[vk_queue_e::dedicated_compute] != -1
              ? queueFamilyProps[queueFamilyMaps[vk_queue_e::dedicated_compute]].queueCount
              : -1,
          queueFamilyMaps[vk_queue_e::dedicated_transfer] != -1
              ? queueFamilyMaps[vk_queue_e::dedicated_transfer]
              : -1,
          queueFamilyMaps[vk_queue_e::dedicated_transfer] != -1
              ? queueFamilyProps[queueFamilyMaps[vk_queue_e::dedicated_transfer]].queueCount
              : -1);
    }

    /// extensions
    int rtPreds = 0;
    constexpr int rtRequiredPreds = 5;
    /// @note the first 5 extensions are required for rt support
    std::vector<const char*> expectedExtensions{
        "VK_KHR_ray_tracing_pipeline",
        "VK_KHR_acceleration_structure",
        "VK_EXT_descriptor_indexing",
        "VK_KHR_buffer_device_address",
        "VK_KHR_deferred_host_operations",
        "VK_KHR_swapchain",
        "VK_KHR_driver_properties",
#ifdef ZS_PLATFORM_OSX
        "VK_KHR_portability_subset",
#endif
        VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,  // "VK_EXT_extended_dynamic_state",
        VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME,
        VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
        VK_KHR_MULTIVIEW_EXTENSION_NAME,
        VK_KHR_MAINTENANCE2_EXTENSION_NAME,
        VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
        VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME};
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

    // query properties 2
    vk::PhysicalDeviceDepthStencilResolveProperties dsResolveProperties{};
    vk::PhysicalDeviceProperties2 devProperties{};
    devProperties.pNext = &dsResolveProperties;
    physicalDevice.getProperties2(&devProperties);

    this->depthStencilResolveProperties = dsResolveProperties;
    this->deviceProperties = devProperties;

    // query features 2
    VkPhysicalDeviceVulkan12Features supportedVk12Features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, nullptr};
    VkPhysicalDeviceFeatures2 devFeatures2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                           &supportedVk12Features};
    dispatcher.vkGetPhysicalDeviceFeatures2(physicalDevice, &devFeatures2);

    this->supportedVk12Features = supportedVk12Features;
    this->supportedDeviceFeatures = devFeatures2;

    vk::PhysicalDeviceFeatures2 features{};

    features.features.fragmentStoresAndAtomics
        = supportedDeviceFeatures.features.fragmentStoresAndAtomics;
    features.features.vertexPipelineStoresAndAtomics
        = supportedDeviceFeatures.features.vertexPipelineStoresAndAtomics;
    features.features.fillModeNonSolid = supportedDeviceFeatures.features.fillModeNonSolid;
    features.features.wideLines = supportedDeviceFeatures.features.wideLines;
    features.features.independentBlend = supportedDeviceFeatures.features.independentBlend;
    features.features.geometryShader = supportedDeviceFeatures.features.geometryShader;
    features.features.tessellationShader = supportedDeviceFeatures.features.tessellationShader;
    this->enabledDeviceFeatures = features;

    vk::DeviceCreateInfo devCI{{},
                               (u32)dqCIs.size(),
                               dqCIs.data(),
                               0,
                               nullptr,
                               (u32)enabledExtensions.size(),
                               enabledExtensions.data()};
    devCI.setPEnabledFeatures(&features.features);

    /// features
    // ref: TU Wien Vulkan Tutorial Ep1
    vk::PhysicalDeviceVulkan12Features vk12Features{};
    // timeline semaphore
    vk12Features.timelineSemaphore = supportedVk12Features.timelineSemaphore;
    //
    vk12Features.descriptorIndexing = supportedVk12Features.descriptorIndexing;
    if (!vk12Features.descriptorIndexing
        && std::find(enabledExtensions.begin(), enabledExtensions.end(),
                     "VK_EXT_descriptor_indexing")
               != enabledExtensions.end())
      vk12Features.descriptorIndexing = VK_TRUE;
    // fmt::print("\n\n\ndescriptor index support: {}\n\n\n",
    // supportedVk12Features.descriptorIndexing);
    vk12Features.bufferDeviceAddress = supportedVk12Features.bufferDeviceAddress;
    // bindless
    vk12Features.descriptorBindingPartiallyBound
        = supportedVk12Features.descriptorBindingPartiallyBound;
    vk12Features.runtimeDescriptorArray = supportedVk12Features.runtimeDescriptorArray;
    // ref: https://zhuanlan.zhihu.com/p/136449475
    vk12Features.descriptorBindingVariableDescriptorCount
        = supportedVk12Features.descriptorBindingVariableDescriptorCount;
    vk12Features.shaderSampledImageArrayNonUniformIndexing
        = supportedVk12Features.shaderSampledImageArrayNonUniformIndexing;

    vk12Features.descriptorBindingUniformBufferUpdateAfterBind
        = supportedVk12Features.descriptorBindingUniformBufferUpdateAfterBind;
    vk12Features.descriptorBindingSampledImageUpdateAfterBind
        = supportedVk12Features.descriptorBindingSampledImageUpdateAfterBind;
    vk12Features.descriptorBindingStorageBufferUpdateAfterBind
        = supportedVk12Features.descriptorBindingStorageBufferUpdateAfterBind;
    vk12Features.descriptorBindingStorageImageUpdateAfterBind
        = supportedVk12Features.descriptorBindingStorageImageUpdateAfterBind;
    this->enabledVk12Features = vk12Features;

    // dynamic states features
    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extendedDynamicStateFeaturesEXT{};
    extendedDynamicStateFeaturesEXT.setExtendedDynamicState(vk::True);
    vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT extendedDynamicState2FeaturesEXT{};
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT extendedDynamicState3FeaturesEXT{};
    extendedDynamicState3FeaturesEXT.setExtendedDynamicState3DepthClampEnable(vk::True);
    extendedDynamicState3FeaturesEXT.setExtendedDynamicState3DepthClipEnable(vk::True);

    extendedDynamicStateFeaturesEXT.setPNext(&extendedDynamicState2FeaturesEXT);
    extendedDynamicState2FeaturesEXT.setPNext(&extendedDynamicState3FeaturesEXT);
    extendedDynamicState3FeaturesEXT.setPNext(&vk12Features);
    // features.setPNext(&extendedDynamicStateFeaturesEXT);

#ifdef ZS_PLATFORM_OSX
    // https://www.lunarg.com/wp-content/uploads/2023/08/Vulkan-Development-in-Apple-Environments-08-09-2023.pdf
    VkPhysicalDevicePortabilitySubsetFeaturesKHR portabilityFeatures{};
    portabilityFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR;
    portabilityFeatures.triangleFans = VK_TRUE;
    portabilityFeatures.pNext = &extendedDynamicStateFeaturesEXT;

    devCI.setPNext(&portabilityFeatures);
#else
    devCI.setPNext(&extendedDynamicStateFeaturesEXT);
#endif

    // ray-tracing feature chaining
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
    asFeatures.accelerationStructure = VK_TRUE;
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rtPipeFeatures{};
    rtPipeFeatures.rayTracingPipeline = VK_TRUE;
    rtPipeFeatures.pNext = &asFeatures;
    if (rtPreds == rtRequiredPreds) vk12Features.pNext = &rtPipeFeatures;

    device = physicalDevice.createDevice(devCI, nullptr, dispatcher);
    dispatcher.init(device);
    ZS_ERROR_IF(!device, fmt::format("Vulkan device [{}] failed initialization!\n", devid));

    VkPhysicalDeviceMemoryProperties tmp;
    dispatcher.vkGetPhysicalDeviceMemoryProperties(physicalDevice, &tmp);
    memoryProperties = tmp;

    /// setup additional resources
    // descriptor pool
    setupDescriptorPool();

    // vma allocator
    {
      VmaVulkanFunctions vulkanFunctions = {};
      vulkanFunctions.vkGetInstanceProcAddr = dispatcher.vkGetInstanceProcAddr;
      vulkanFunctions.vkGetDeviceProcAddr = dispatcher.vkGetDeviceProcAddr;

      VmaAllocatorCreateInfo allocatorCreateInfo = {};
      allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
      allocatorCreateInfo.physicalDevice = physicalDevice;
      allocatorCreateInfo.device = device;
      allocatorCreateInfo.instance = instance;
      allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;

      vmaCreateAllocator(&allocatorCreateInfo, &this->defaultAllocator);
    }

    /// display info
    fmt::print(
        "\t[InitInfo -- Dev Property] Vulkan device [{}] name: {}."
        "\n\t\t(Graphics/Compute/Transfer) queue family index: {}, {}, {}. Ray-tracing support: "
        "{}. Bindless support: {}. Timeline semaphore support: {}"
        "\n\tEnabled the following device tensions ({} in total):",
        devid, devProps.deviceName.data(), queueFamilyIndices[vk_queue_e::graphics],
        queueFamilyIndices[vk_queue_e::compute], queueFamilyIndices[vk_queue_e::transfer],
        rtPreds == rtRequiredPreds, supportBindless(), vk12Features.timelineSemaphore,
        enabledExtensions.size());
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

  void VulkanContext::destructDescriptorPool() {
    /// clear resources
    if (supportBindless() && bindlessDescriptorPool != VK_NULL_HANDLE) {
      // descriptor pool resources
      bindlessDescriptorSet = VK_NULL_HANDLE;

      device.destroyDescriptorSetLayout(bindlessDescriptorSetLayout, nullptr, dispatcher);
      bindlessDescriptorSetLayout = VK_NULL_HANDLE;

      device.resetDescriptorPool(bindlessDescriptorPool, vk::DescriptorPoolResetFlags{},
                                 dispatcher);
      device.destroyDescriptorPool(bindlessDescriptorPool, nullptr, dispatcher);
      bindlessDescriptorPool = VK_NULL_HANDLE;
    }
    if (defaultDescriptorPool != VK_NULL_HANDLE) {
      device.resetDescriptorPool(defaultDescriptorPool, vk::DescriptorPoolResetFlags{}, dispatcher);
      device.destroyDescriptorPool(defaultDescriptorPool, nullptr, dispatcher);
      defaultDescriptorPool = VK_NULL_HANDLE;
    }
  }

  void VulkanContext::setupDescriptorPool() {
    /// pool
    std::array<vk::DescriptorPoolSize, vk_descriptor_e::num_descriptor_types> poolSizes;
    poolSizes[vk_descriptor_e::uniform] = vk::DescriptorPoolSize()
                                              .setDescriptorCount(num_max_default_resources)
                                              .setType(vk::DescriptorType::eUniformBufferDynamic);

    poolSizes[vk_descriptor_e::image_sampler]
        = vk::DescriptorPoolSize()
              .setDescriptorCount(num_max_default_resources)
              .setType(vk::DescriptorType::eCombinedImageSampler);

    poolSizes[vk_descriptor_e::storage] = vk::DescriptorPoolSize()
                                              .setDescriptorCount(num_max_default_resources)
                                              .setType(vk::DescriptorType::eStorageBuffer);

    poolSizes[vk_descriptor_e::storage_image] = vk::DescriptorPoolSize()
                                                    .setDescriptorCount(num_max_default_resources)
                                                    .setType(vk::DescriptorType::eStorageImage);
    poolSizes[vk_descriptor_e::input_attachment]
        = vk::DescriptorPoolSize()
              .setDescriptorCount(num_max_default_resources)
              .setType(vk::DescriptorType::eInputAttachment);
    vk::DescriptorPoolCreateFlags flag = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    defaultDescriptorPool
        = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{}
                                          .setPoolSizeCount(poolSizes.size())
                                          .setPPoolSizes(poolSizes.data())
                                          .setMaxSets(num_max_default_resources * poolSizes.size())
                                          .setFlags(flag),
                                      nullptr, dispatcher);

    bindlessDescriptorPool = VK_NULL_HANDLE;
    bindlessDescriptorSetLayout = VK_NULL_HANDLE;
    bindlessDescriptorSet = VK_NULL_HANDLE;

    if (!supportBindless()) return;
    flag |= vk::DescriptorPoolCreateFlagBits::eUpdateAfterBindEXT;

    {
      std::array<vk::DescriptorPoolSize, vk_descriptor_e::num_descriptor_types> bindlessPoolSizes;
      bindlessPoolSizes[vk_descriptor_e::uniform]
          = vk::DescriptorPoolSize()
                .setDescriptorCount(num_max_bindless_resources)
                .setType(vk::DescriptorType::eUniformBuffer);
      bindlessPoolSizes[vk_descriptor_e::image_sampler]
          = vk::DescriptorPoolSize()
                .setDescriptorCount(num_max_bindless_resources)
                .setType(vk::DescriptorType::eCombinedImageSampler);

      bindlessPoolSizes[vk_descriptor_e::storage]
          = vk::DescriptorPoolSize()
                .setDescriptorCount(num_max_bindless_resources)
                .setType(vk::DescriptorType::eStorageBuffer);

      bindlessPoolSizes[vk_descriptor_e::storage_image]
          = vk::DescriptorPoolSize()
                .setDescriptorCount(num_max_bindless_resources)
                .setType(vk::DescriptorType::eStorageImage);

      bindlessPoolSizes[vk_descriptor_e::input_attachment]
          = vk::DescriptorPoolSize()
                .setDescriptorCount(num_max_bindless_resources)
                .setType(vk::DescriptorType::eInputAttachment);
      bindlessDescriptorPool = device.createDescriptorPool(
          vk::DescriptorPoolCreateInfo{}
              .setPoolSizeCount(bindlessPoolSizes.size())
              .setPPoolSizes(bindlessPoolSizes.data())
              .setMaxSets(num_max_bindless_resources * bindlessPoolSizes.size())
              .setFlags(flag),
          nullptr, dispatcher);
    }

    /// set layout
    std::array<vk::DescriptorSetLayoutBinding, vk_descriptor_e::num_descriptor_types> bindings;
    auto& uniformBinding = bindings[vk_descriptor_e::uniform];
    uniformBinding = vk::DescriptorSetLayoutBinding{}
                         .setBinding(bindless_texture_binding)
                         .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                         .setDescriptorCount(num_max_bindless_resources)
                         .setStageFlags(vk::ShaderStageFlagBits::eAll);
    auto& imageSamplerBinding = bindings[vk_descriptor_e::image_sampler];
    imageSamplerBinding = vk::DescriptorSetLayoutBinding{}
                              .setBinding(bindless_texture_binding + 1)
                              .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                              .setDescriptorCount(num_max_bindless_resources)
                              .setStageFlags(vk::ShaderStageFlagBits::eAll);
    auto& storageBinding = bindings[vk_descriptor_e::storage];
    storageBinding = vk::DescriptorSetLayoutBinding{}
                         .setBinding(bindless_texture_binding + 2)
                         .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                         .setDescriptorCount(num_max_bindless_resources)
                         .setStageFlags(vk::ShaderStageFlagBits::eAll);
    auto& storageImageBinding = bindings[vk_descriptor_e::storage_image];
    storageImageBinding = vk::DescriptorSetLayoutBinding{}
                              .setBinding(bindless_texture_binding + 3)
                              .setDescriptorType(vk::DescriptorType::eStorageImage)
                              .setDescriptorCount(num_max_bindless_resources)
                              .setStageFlags(vk::ShaderStageFlagBits::eAll);
    auto& inputAttachmentBinding = bindings[vk_descriptor_e::input_attachment];
    inputAttachmentBinding = vk::DescriptorSetLayoutBinding{}
                                 .setBinding(bindless_texture_binding + 4)
                                 .setDescriptorType(vk::DescriptorType::eInputAttachment)
                                 .setDescriptorCount(num_max_bindless_resources)
                                 .setStageFlags(vk::ShaderStageFlagBits::eFragment);

    vk::DescriptorBindingFlags bindlessFlag = vk::DescriptorBindingFlagBits::ePartiallyBound
                                              | vk::DescriptorBindingFlagBits::eUpdateAfterBind;
    std::array<vk::DescriptorBindingFlags, vk_descriptor_e::num_descriptor_types> bindingFlags;
    for (auto& flag : bindingFlags) flag = bindlessFlag;
    bindingFlags[vk_descriptor_e::input_attachment]
        = vk::DescriptorBindingFlagBits::ePartiallyBound;
    auto extendedInfo
        = vk::DescriptorSetLayoutBindingFlagsCreateInfo{}.setBindingFlags(bindingFlags);

    bindlessDescriptorSetLayout = device.createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{}
            .setFlags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPoolEXT)
            .setBindingCount(bindings.size())
            .setPBindings(bindings.data())
            .setPNext(&extendedInfo),
        nullptr, dispatcher);

    /// set
    bindlessDescriptorSet
        = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}
                                            .setDescriptorPool(bindlessDescriptorPool)
                                            .setPSetLayouts(&bindlessDescriptorSetLayout)
                                            .setDescriptorSetCount(1))[0];
  }

  ExecutionContext& VulkanContext::env() {
    WorkerEnvs::iterator workerIter;
    ContextEnvs::iterator iter;
    auto& g_mtx = Vulkan::instance().mutex<Mutex>();
    g_mtx.lock();
    bool tag;
    std::tie(workerIter, tag) = Vulkan::instance().working_contexts<WorkerEnvs>().try_emplace(
        std::this_thread::get_id(), ContextEnvs{});
    std::tie(iter, tag) = workerIter->second.try_emplace(devid, *this);
    g_mtx.unlock();
    return iter->second;
  }
  u32 check_current_working_contexts() {
    return Vulkan::instance().working_contexts<WorkerEnvs>().size();
  }

  ///
  /// builders
  ///
  SwapchainBuilder& VulkanContext::swapchain(vk::SurfaceKHR surface, bool reset) {
    if ((!swapchainBuilder || reset
         || ((SwapchainBuilder*)swapchainBuilder)->getSurface() != surface)
        && surface != VK_NULL_HANDLE)
      swapchainBuilder.reset(new SwapchainBuilder(*this, surface));
    if (swapchainBuilder)
      return *(SwapchainBuilder*)swapchainBuilder;
    else
      throw std::runtime_error(
          "swapchain builder of the vk context must be initialized by a surface first before use");
  }
  PipelineBuilder VulkanContext::pipeline() { return PipelineBuilder{*this}; }
  RenderPassBuilder VulkanContext::renderpass() { return RenderPassBuilder(*this); }
  DescriptorSetLayoutBuilder VulkanContext::setlayout() {
    return DescriptorSetLayoutBuilder{*this};
  }

  image_handle_t VulkanContext::registerImage(const VkTexture& img) {
    if (!supportBindless()) return (image_handle_t)-1;
    image_handle_t ret = registeredImages.size();
    registeredImages.push_back(&img);

    vk::DescriptorImageInfo imageInfo{};
    imageInfo.sampler = img.sampler;
    imageInfo.imageView = (vk::ImageView)img.image.get();
    imageInfo.imageLayout = img.imageLayout;

    // if ((vk::ImageView)img.image.get() == VK_NULL_HANDLE)
    //   throw std::runtime_error("the registered texture image view handle is null\n");

    vk::WriteDescriptorSet write{};
    write.dstSet = bindlessDescriptorSet;
    write.descriptorCount = 1;
    write.dstArrayElement = ret;
    write.pImageInfo = &imageInfo;

    std::vector<vk::WriteDescriptorSet> writes{};
    if ((img.image.get().usage & vk::ImageUsageFlagBits::eSampled)
        == vk::ImageUsageFlagBits::eSampled) {
      write.descriptorType = vk::DescriptorType::eCombinedImageSampler;
      write.dstBinding = (u32)vk_descriptor_e::image_sampler;
      writes.push_back(write);
    }
    if ((img.image.get().usage & vk::ImageUsageFlagBits::eStorage)
        == vk::ImageUsageFlagBits::eStorage) {
      write.descriptorType = vk::DescriptorType::eStorageImage;
      write.dstBinding = (u32)vk_descriptor_e::storage_image;
      writes.push_back(write);
    }

    device.updateDescriptorSets(writes.size(), writes.data(), 0, nullptr, dispatcher);
    return ret;
  }

  buffer_handle_t VulkanContext::registerBuffer(const Buffer& buffer) {
    buffer_handle_t ret = registeredBuffers.size();
    registeredBuffers.push_back(&buffer);

    vk::DescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = buffer.getSize();

    vk::WriteDescriptorSet write{};
    write.dstSet = bindlessDescriptorSet;
    write.descriptorCount = 1;
    write.dstArrayElement = ret;
    write.pBufferInfo = &bufferInfo;

    std::vector<vk::WriteDescriptorSet> writes{};
    if ((buffer.usageFlags & vk::BufferUsageFlagBits::eUniformBuffer)
        == vk::BufferUsageFlagBits::eUniformBuffer) {
      write.descriptorType = vk::DescriptorType::eUniformBuffer;
      write.dstBinding = (u32)vk_descriptor_e::uniform;
      writes.push_back(write);
    }
    if ((buffer.usageFlags & vk::BufferUsageFlagBits::eStorageBuffer)
        == vk::BufferUsageFlagBits::eStorageBuffer) {
      write.descriptorType = vk::DescriptorType::eStorageBuffer;
      write.dstBinding = (u32)vk_descriptor_e::storage;
      writes.push_back(write);
    }

    device.updateDescriptorSets(writes.size(), writes.data(), 0, nullptr, dispatcher);
    return ret;
  }

  Buffer VulkanContext::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                     vk::MemoryPropertyFlags props) {
    Buffer buffer(*this);

    vk::BufferCreateInfo bufCI{};
    bufCI.setUsage(usage);
    bufCI.setSize(size);
    bufCI.setSharingMode(vk::SharingMode::eExclusive);
    auto buf = device.createBuffer(bufCI, nullptr, dispatcher);

#if ZS_VULKAN_USE_VMA
    auto bufferReqs = vk::BufferMemoryRequirementsInfo2{}.setBuffer(buf);
    auto dedicatedReqs = vk::MemoryDedicatedRequirements{};
    dedicatedReqs.prefersDedicatedAllocation = true;
    auto memReqs2 = vk::MemoryRequirements2{};
    memReqs2.pNext = &dedicatedReqs;

    device.getBufferMemoryRequirements2(&bufferReqs, &memReqs2, dispatcher);

    auto& memRequirements = memReqs2.memoryRequirements;

    VmaAllocationCreateInfo vmaAllocCI = {};
    if (dedicatedReqs.requiresDedicatedAllocation)
      vmaAllocCI.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    vmaAllocCI.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    vmaAllocCI.usage = VMA_MEMORY_USAGE_UNKNOWN;
    // vmaAllocCI.usage = vk_to_vma_memory_usage(props);  // deprecated
    vmaAllocCI.requiredFlags = static_cast<VkMemoryPropertyFlags>(props);
    vmaAllocCI.priority = 1.f;

    VmaAllocationInfo allocationDetail;
    VmaAllocation allocation = nullptr;
    VkResult result
        = vmaAllocateMemory(allocator(), reinterpret_cast<VkMemoryRequirements*>(&memRequirements),
                            &vmaAllocCI, &allocation, &allocationDetail);
    if (result != VK_SUCCESS)
      throw std::runtime_error(fmt::format("buffer allocation of {} bytes failed!", size));

    device.bindBufferMemory(buf, allocationDetail.deviceMemory, allocationDetail.offset,
                            dispatcher);
#else
    vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buf, dispatcher);
    u32 memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, props);
    vk::MemoryAllocateInfo allocInfo{memRequirements.size, memoryTypeIndex};
    auto mem = device.allocateMemory(allocInfo, nullptr, dispatcher);

    device.bindBufferMemory(buf, mem, 0, dispatcher);
#endif

    buffer.size = size;
    buffer.usageFlags = usage;
    buffer.alignment = memRequirements.alignment;
    buffer.buffer = buf;

#if ZS_VULKAN_USE_VMA
    buffer.allocation = allocation;
#else
    VkMemory memory{*this};
    memory.mem = mem;
    memory.memSize = memRequirements.size;
    memory.memoryPropertyFlags = memoryProperties.memoryTypes[memoryTypeIndex].propertyFlags;
    buffer.pmem = std::make_shared<VkMemory>(std::move(memory));
#endif
    return buffer;
  }
  Buffer VulkanContext::createStagingBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage) {
    return createBuffer(
        size, usage,
        vk::MemoryPropertyFlagBits::eHostVisible /* | vk::MemoryPropertyFlagBits::eHostCoherent*/);
  }

  ImageSampler VulkanContext::createSampler(const vk::SamplerCreateInfo& samplerCI) {
    ImageSampler ret{*this};
    ret.sampler = device.createSampler(samplerCI, nullptr, dispatcher);
    return ret;
  }
  ImageSampler VulkanContext::createDefaultSampler() {
    return createSampler(vk::SamplerCreateInfo{}
                             .setMaxAnisotropy(1.f)
                             .setMagFilter(vk::Filter::eLinear)
                             .setMinFilter(vk::Filter::eLinear)
                             .setMipmapMode(vk::SamplerMipmapMode::eLinear)
                             .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
                             .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
                             .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
                             .setBorderColor(vk::BorderColor::eFloatOpaqueWhite));
  }

  Image VulkanContext::createImage(vk::ImageCreateInfo imageCI, vk::MemoryPropertyFlags props,
                                   bool createView) {
    Image image{*this};
    image.usage = imageCI.usage;
    image.extent = imageCI.extent;
    image.mipLevels = imageCI.mipLevels;

    auto img = device.createImage(imageCI, nullptr, dispatcher);

#if ZS_VULKAN_USE_VMA
    auto imageReqs = vk::ImageMemoryRequirementsInfo2{}.setImage(img);
    auto dedicatedReqs = vk::MemoryDedicatedRequirements{};
    dedicatedReqs.prefersDedicatedAllocation = true;
    auto memReqs2 = vk::MemoryRequirements2{};
    memReqs2.pNext = &dedicatedReqs;

    device.getImageMemoryRequirements2(&imageReqs, &memReqs2, dispatcher);

    auto& memRequirements = memReqs2.memoryRequirements;

    VmaAllocationCreateInfo vmaAllocCI = {};
    if (dedicatedReqs.requiresDedicatedAllocation)
      vmaAllocCI.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    vmaAllocCI.usage = VMA_MEMORY_USAGE_UNKNOWN;
    // vmaAllocCI.usage = vk_to_vma_memory_usage(props);  // deprecated
    vmaAllocCI.requiredFlags = static_cast<VkMemoryPropertyFlags>(props);
    vmaAllocCI.priority = 1.f;

    VmaAllocationInfo allocationDetail;
    VmaAllocation allocation = nullptr;
    VkResult result
        = vmaAllocateMemory(allocator(), reinterpret_cast<VkMemoryRequirements*>(&memRequirements),
                            &vmaAllocCI, &allocation, &allocationDetail);
    if (result != VK_SUCCESS)
      throw std::runtime_error(fmt::format("image allocation of dim [{}, {}] failed!",
                                           imageCI.extent.width, imageCI.extent.height));

    device.bindImageMemory(img, allocationDetail.deviceMemory, allocationDetail.offset, dispatcher);
#else
    vk::MemoryRequirements memRequirements = device.getImageMemoryRequirements(img, dispatcher);
    u32 memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, props);
    vk::MemoryAllocateInfo allocInfo{memRequirements.size, memoryTypeIndex};
    auto mem = device.allocateMemory(allocInfo, nullptr, dispatcher);

    device.bindImageMemory(img, mem, 0, dispatcher);
#endif

    image.image = img;
#if ZS_VULKAN_USE_VMA
    image.allocation = allocation;
#else
    VkMemory memory{*this};
    memory.mem = mem;
    memory.memSize = memRequirements.size;
    memory.memoryPropertyFlags = memoryProperties.memoryTypes[memoryTypeIndex].propertyFlags;
    image.pmem = std::make_shared<VkMemory>(std::move(memory));
#endif
    if (createView) {
      image.pview = device.createImageView(
          vk::ImageViewCreateInfo{}
              .setImage(img)
              .setPNext(nullptr)
              .setViewType(vk::ImageViewType::e2D)
              .setFormat(imageCI.format)
              .setSubresourceRange(vk::ImageSubresourceRange{
                  is_depth_stencil_format(imageCI.format) ? vk::ImageAspectFlagBits::eDepth
                                                          : vk::ImageAspectFlagBits::eColor,
                  0, 1 /*VK_REMAINING_MIP_LEVELS*/, 0, 1
                  /*VK_REMAINING_ARRAY_LAYERS*/}),
          nullptr, dispatcher);
    }
    return image;
  }
  Image VulkanContext::create2DImage(const vk::Extent2D& dim, vk::Format format,
                                     vk::ImageUsageFlags usage, vk::MemoryPropertyFlags props,
                                     bool mipmaps, bool createView, bool enableTransfer,
                                     vk::SampleCountFlagBits sampleBits) {
    return createImage(vk::ImageCreateInfo{}
                           .setImageType(vk::ImageType::e2D)
                           .setFormat(format)
                           .setExtent({dim.width, dim.height, (u32)1})
                           .setMipLevels((mipmaps ? get_num_mip_levels(dim) : 1))
                           .setArrayLayers(1)
                           .setUsage(enableTransfer ? (usage | vk::ImageUsageFlagBits::eTransferSrc
                                                       | vk::ImageUsageFlagBits::eTransferDst)
                                                    : usage)
                           .setSamples(sampleBits)
                           //.setTiling(vk::ImageTiling::eOptimal)
                           .setSharingMode(vk::SharingMode::eExclusive),
                       props, createView);
  }
  Image VulkanContext::createOptimal2DImage(const vk::Extent2D& dim, vk::Format format,
                                            vk::ImageUsageFlags usage,
                                            vk::MemoryPropertyFlags props, bool mipmaps,
                                            bool createView, bool enableTransfer,
                                            vk::SampleCountFlagBits sampleBits) {
    return createImage(vk::ImageCreateInfo{}
                           .setImageType(vk::ImageType::e2D)
                           .setFormat(format)
                           .setExtent({dim.width, dim.height, (u32)1})
                           .setMipLevels((mipmaps ? get_num_mip_levels(dim) : 1))
                           .setArrayLayers(1)
                           .setUsage(enableTransfer ? (usage | vk::ImageUsageFlagBits::eTransferSrc
                                                       | vk::ImageUsageFlagBits::eTransferDst)
                                                    : usage)
                           .setSamples(sampleBits)
                           .setTiling(vk::ImageTiling::eOptimal)
                           .setSharingMode(vk::SharingMode::eExclusive),
                       props, createView);
  }
  Image VulkanContext::createInputAttachment(const vk::Extent2D& dim, vk::Format format,
                                             vk::ImageUsageFlags usage, bool enableTransfer) {
    usage |= vk::ImageUsageFlagBits::eInputAttachment;
    return createImage(vk::ImageCreateInfo{}
                           .setImageType(vk::ImageType::e2D)
                           .setFormat(format)
                           .setExtent({dim.width, dim.height, (u32)1})
                           .setMipLevels(1)
                           .setArrayLayers(1)
                           .setUsage(enableTransfer ? (usage | vk::ImageUsageFlagBits::eTransferSrc
                                                       | vk::ImageUsageFlagBits::eTransferDst)
                                                    : usage)
                           .setSamples(vk::SampleCountFlagBits::e1)
                           // .setTiling(vk::ImageTiling::eOptimal)
                           .setSharingMode(vk::SharingMode::eExclusive),
                       vk::MemoryPropertyFlagBits::eDeviceLocal, true);
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

  VkCommand VulkanContext::createCommandBuffer(vk_cmd_usage_e usage, vk_queue_e queueFamily,
                                               bool begin) {
    auto& pool = env().pools(queueFamily);
    auto cmd = pool.createCommandBuffer(vk::CommandBufferLevel::ePrimary, begin,
                                        /*inheritance info*/ nullptr, usage);
    return VkCommand{pool, cmd, usage};
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
  void VulkanContext::writeDescriptorSet(const vk::DescriptorBufferInfo& bufferInfo,
                                         vk::DescriptorSet dstSet, vk::DescriptorType type,
                                         u32 binding, u32 dstArrayNo) {
    auto write = vk::WriteDescriptorSet{}
                     .setDescriptorType(type)
                     .setDstSet(dstSet)
                     .setDstBinding(binding)
                     .setDstArrayElement(dstArrayNo)
                     .setDescriptorCount((u32)1)
                     .setPBufferInfo(&bufferInfo);
    device.updateDescriptorSets(1, &write, 0, nullptr, dispatcher);
  }
  void VulkanContext::writeDescriptorSet(const vk::DescriptorImageInfo& imageInfo,
                                         vk::DescriptorSet dstSet, vk::DescriptorType type,
                                         u32 binding, u32 dstArrayNo) {
    auto write = vk::WriteDescriptorSet{}
                     .setDescriptorType(type)
                     .setDstSet(dstSet)
                     .setDstBinding(binding)
                     .setDstArrayElement(dstArrayNo)
                     .setDescriptorCount((u32)1)
                     .setPImageInfo(&imageInfo);
    device.updateDescriptorSets(1, &write, 0, nullptr, dispatcher);
  }

  ///
  ///
  /// working context (CmdContext)
  ///
  ///
  ExecutionContext::ExecutionContext(VulkanContext& ctx)
      : ctx{ctx}, poolFamilies(ctx.numDistinctQueueFamilies()) {
    for (const auto& [no, family, queueFamilyIndex] :
         enumerate(poolFamilies, ctx.uniqueQueueFamilyIndices)) {
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
      /// setup preset primary command buffers (reuse)
      family.primaryCmd
          = new VkCommand(family,
                          ctx.device.allocateCommandBuffers(
                              vk::CommandBufferAllocateInfo{
                                  family.resetPool, vk::CommandBufferLevel::ePrimary, (u32)1},
                              ctx.dispatcher)[0],
                          vk_cmd_usage_e::reset);
      family.fence = new Fence(ctx, true);

      //
      family.queue = ctx.device.getQueue(queueFamilyIndex, 0, ctx.dispatcher);
      family.allQueues.resize(ctx.getQueueFamilyPropertyByIndex(no).queueCount);
      for (int i = 0; i < family.allQueues.size(); ++i)
        family.allQueues[i] = ctx.device.getQueue(queueFamilyIndex, i, ctx.dispatcher);
      family.pctx = &ctx;
    }
  }
  ExecutionContext::~ExecutionContext() {
    for (auto& family : poolFamilies) {
      /// @brief clear secondary command buffers before destroying command pools
      if (family.primaryCmd) {
        delete family.primaryCmd;
        family.primaryCmd = nullptr;
      }
      if (family.fence) {
        delete family.fence;
        family.fence = nullptr;
      }
      for (auto& ptr : family.secondaryCmds)
        if (ptr) delete ptr;
      family.secondaryCmds.clear();
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
  VkCommand ExecutionContext::PoolFamily::createVkCommand(vk_cmd_usage_e usage, bool begin) {
    const auto& cmdPool = cmdpool(usage);
    std::vector<vk::CommandBuffer> cmd = pctx->device.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo{cmdPool, vk::CommandBufferLevel::ePrimary, (u32)1},
        pctx->dispatcher);
    VkCommand ret{*this, cmd[0], usage};
    if (begin) ret.begin();
    return ret;
  }
  VkCommand& ExecutionContext::PoolFamily::acquireSecondaryVkCommand() {
    auto cmdPtr
        = new VkCommand(*this,
                        createCommandBuffer(vk::CommandBufferLevel::eSecondary, false,
                                            /*inheritance info*/ nullptr, vk_cmd_usage_e::reset),
                        vk_cmd_usage_e::reset);
    secondaryCmds.emplace_back(cmdPtr);
    secondaryCmdHandles.emplace_back(*secondaryCmds.back());
    return *secondaryCmds.back();
  }
  VkCommand& ExecutionContext::PoolFamily::acquireSecondaryVkCommand(int k) {
    while (k >= secondaryCmds.size()) acquireSecondaryVkCommand();
    return *secondaryCmds[k];
  }

  const VkCommand& ExecutionContext::PoolFamily::retrieveSecondaryVkCommand(int k) const {
    assert(k >= 0 && k < secondaryCmds.size());
    return *secondaryCmds[k];
  }

  std::vector<vk::CommandBuffer> ExecutionContext::PoolFamily::retrieveSecondaryVkCommands(
      int n) const {
    if (n < 0 || n >= secondaryCmdHandles.size()) return secondaryCmdHandles;
    std::vector<vk::CommandBuffer> ret(n);
    for (int i = 0; i < n; ++i) ret.push_back(secondaryCmdHandles[i]);
    return ret;
  }

}  // namespace zs