// vulkan memory allocator impl
#include <vulkan/vulkan_core.h>
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
    /// clear execution resources
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
    destructDescriptorPool();

    vmaDestroyAllocator(defaultAllocator);
    defaultAllocator = 0;  // ref: nvpro-core

    /// destroy logical device
    device.destroy(nullptr, dispatcher);
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
    auto queueFamilyProps = physicalDevice.getQueueFamilyProperties();
    for (auto& queueFamilyIndex : queueFamilyIndices) queueFamilyIndex = -1;
    for (auto& queueFamilyMap : queueFamilyMaps) queueFamilyMap = -1;
    for (int i = 0; i != queueFamilyProps.size(); ++i) {
      auto& q = queueFamilyProps[i];
      if (queueFamilyIndices[vk_queue_e::graphics] == -1
          && (q.queueFlags & vk::QueueFlagBits::eGraphics)) {
        queueFamilyIndices[vk_queue_e::graphics] = i;
        ZS_WARN_IF(!(q.queueFlags & vk::QueueFlagBits::eTransfer),
                   "the selected graphics queue family cannot transfer!");
      }
      if (queueFamilyIndices[vk_queue_e::compute] == -1
          && (q.queueFlags & vk::QueueFlagBits::eCompute))
        queueFamilyIndices[vk_queue_e::compute] = i;
      if (queueFamilyIndices[vk_queue_e::transfer] == -1
          && (q.queueFlags & vk::QueueFlagBits::eTransfer))
        queueFamilyIndices[vk_queue_e::transfer] = i;
    }
    ZS_ERROR_IF(queueFamilyIndices[vk_queue_e::graphics] == -1,
                "graphics queue family does not exist!");
    fmt::print("selected queue family [{}] for graphics! (compute: {}, transfer: {})\n",
               queueFamilyIndices[vk_queue_e::graphics], queueFamilyIndices[vk_queue_e::compute],
               queueFamilyIndices[vk_queue_e::transfer]);

    std::set<u32> uniqueQueueFamilyIndices{(u32)queueFamilyIndices[vk_queue_e::graphics],
                                           (u32)queueFamilyIndices[vk_queue_e::compute],
                                           (u32)queueFamilyIndices[vk_queue_e::transfer]};
    this->uniqueQueueFamilyIndices.reserve(uniqueQueueFamilyIndices.size());
    std::vector<vk::DeviceQueueCreateInfo> dqCIs(uniqueQueueFamilyIndices.size());
    float priority = 1.f;
    {
      u32 i = 0;
      for (auto index : uniqueQueueFamilyIndices) {
        auto& dqCI = dqCIs[i];
        this->uniqueQueueFamilyIndices.push_back(index);
        dqCI.setQueueFamilyIndex(index).setQueueCount(1).setPQueuePriorities(&priority);

        if (queueFamilyIndices[vk_queue_e::graphics] == index) queueFamilyMaps[0] = i;
        if (queueFamilyIndices[vk_queue_e::compute] == index) queueFamilyMaps[1] = i;
        if (queueFamilyIndices[vk_queue_e::transfer] == index) queueFamilyMaps[2] = i;
        i++;
      }
      fmt::print("queue family maps (graphics: {}, compute: {}, transfer: {})\n",
                 queueFamilyMaps[0], queueFamilyMaps[1], queueFamilyMaps[2]);
    }

    /// extensions
    int rtPreds = 0;
    constexpr int rtRequiredPreds = 5;
    /// @note the first 5 extensions are required for rt support
    std::vector<const char*> expectedExtensions{
        "VK_KHR_ray_tracing_pipeline",     "VK_KHR_acceleration_structure",
        "VK_EXT_descriptor_indexing",      "VK_KHR_buffer_device_address",
        "VK_KHR_deferred_host_operations", "VK_KHR_swapchain",
        "VK_KHR_driver_properties"};
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

    // query features 2
    VkPhysicalDeviceVulkan12Features supportedVk12Features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, nullptr};
    VkPhysicalDeviceFeatures2 devFeatures2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                           &supportedVk12Features};
    dispatcher.vkGetPhysicalDeviceFeatures2(physicalDevice, &devFeatures2);

    this->supportedVk12Features = supportedVk12Features;
    this->supportedDeviceFeatures = devFeatures2;

    vk::PhysicalDeviceFeatures2 features;

    features.features.fillModeNonSolid = supportedDeviceFeatures.features.fillModeNonSolid;
    features.features.wideLines = supportedDeviceFeatures.features.wideLines;
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
    vk12Features.descriptorIndexing = supportedVk12Features.descriptorIndexing;
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

    devCI.pNext = &vk12Features;

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
        "{}. Bindless support: {}."
        "\n\tEnabled the following device tensions ({} in total):",
        devid, devProps.deviceName, queueFamilyIndices[vk_queue_e::graphics],
        queueFamilyIndices[vk_queue_e::compute], queueFamilyIndices[vk_queue_e::transfer],
        rtPreds == rtRequiredPreds, supportBindless(), enabledExtensions.size());
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

    vk::DescriptorBindingFlags bindlessFlag = vk::DescriptorBindingFlagBits::ePartiallyBound
                                              | vk::DescriptorBindingFlagBits::eUpdateAfterBind;
    std::array<vk::DescriptorBindingFlags, vk_descriptor_e::num_descriptor_types> bindingFlags;
    for (auto& flag : bindingFlags) flag = bindlessFlag;
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
    g_mtx.lock();
    bool tag;
    std::tie(workerIter, tag)
        = g_workingContexts.try_emplace(std::this_thread::get_id(), ContextEnvs{});
    std::tie(iter, tag) = workerIter->second.try_emplace(devid, *this);
    g_mtx.unlock();
    return iter->second;
  }
  u32 check_current_working_contexts() { return g_workingContexts.size(); }

  ///
  /// builders
  ///
  SwapchainBuilder& VulkanContext::swapchain(vk::SurfaceKHR surface, bool reset) {
    if ((!swapchainBuilder || reset || swapchainBuilder->getSurface() != surface)
        && surface != VK_NULL_HANDLE)
      swapchainBuilder.reset(new SwapchainBuilder(*this, surface));
    if (swapchainBuilder)
      return *swapchainBuilder;
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
    image_handle_t ret = registeredImages.size();
    registeredImages.push_back(&img);

    vk::DescriptorImageInfo imageInfo{};
    imageInfo.sampler = img.sampler;
    imageInfo.imageView = (vk::ImageView)img.image.get();
    imageInfo.imageLayout = img.imageLayout;

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
    auto memReqs2 = vk::MemoryRequirements2{};
    memReqs2.pNext = &dedicatedReqs;

    device.getBufferMemoryRequirements2(&bufferReqs, &memReqs2, dispatcher);

    auto& memRequirements = memReqs2.memoryRequirements;

    VmaAllocationCreateInfo vmaAllocCI = {};
    if (dedicatedReqs.requiresDedicatedAllocation)
      vmaAllocCI.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    vmaAllocCI.usage = vk_to_vma_memory_usage(props);
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

  Image VulkanContext::createImage(vk::ImageCreateInfo imageCI, vk::MemoryPropertyFlags props,
                                   bool createView) {
    Image image{*this};
    image.usage = imageCI.usage;

    auto img = device.createImage(imageCI, nullptr, dispatcher);

#if ZS_VULKAN_USE_VMA
    auto imageReqs = vk::ImageMemoryRequirementsInfo2{}.setImage(img);
    auto dedicatedReqs = vk::MemoryDedicatedRequirements{};
    auto memReqs2 = vk::MemoryRequirements2{};
    memReqs2.pNext = &dedicatedReqs;

    device.getImageMemoryRequirements2(&imageReqs, &memReqs2, dispatcher);

    auto& memRequirements = memReqs2.memoryRequirements;

    VmaAllocationCreateInfo vmaAllocCI = {};
    if (dedicatedReqs.requiresDedicatedAllocation)
      vmaAllocCI.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    vmaAllocCI.usage = vk_to_vma_memory_usage(props);
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
                  is_depth_format(imageCI.format) ? vk::ImageAspectFlagBits::eDepth
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
                           .setTiling(vk::ImageTiling::eOptimal)
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
  void VulkanContext::writeDescriptor(const vk::DescriptorBufferInfo& bufferInfo,
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
  void VulkanContext::writeDescriptor(const vk::DescriptorImageInfo& imageInfo,
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
  VkCommand ExecutionContext::PoolFamily::createVkCommand(vk_cmd_usage_e usage, bool begin) {
    const auto& cmdPool = cmdpool(usage);
    std::vector<vk::CommandBuffer> cmd = pctx->device.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo{cmdPool, vk::CommandBufferLevel::ePrimary, (u32)1},
        pctx->dispatcher);
    VkCommand ret{*this, cmd[0], usage};
    if (begin) ret.begin();
    return ret;
  }

}  // namespace zs