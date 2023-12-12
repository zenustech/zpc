#include "zensim/vulkan/VkContext.hpp"

namespace zs {

  bool VulkanContext::retrieveQueue(vk::Queue &q, vk_queue_e e, u32 i) const noexcept {
    auto index = queueFamilyIndices[e];
    if (index != -1) {
      q = device.getQueue(index, i, dispatcher);
      return true;
    }
    return false;
  }

  bool VulkanContext::supportSurface(vk::SurfaceKHR surface) const {
    if (queueFamilyIndices[vk_queue_e::graphics] == -1) return false;
    return physicalDevice.getSurfaceSupportKHR(queueFamilyIndices[vk_queue_e::graphics], surface,
                                               dispatcher);
  }

  u32 VulkanContext::findMemoryType(u32 memoryTypeBits, vk::MemoryPropertyFlags properties) const {
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

  vk::Format VulkanContext::findSupportedFormat(const std::vector<vk::Format> &candidates,
                                                vk::ImageTiling tiling,
                                                vk::FormatFeatureFlags features) const {
    for (vk::Format format : candidates) {
      VkFormatProperties props;
      dispatcher.vkGetPhysicalDeviceFormatProperties(physicalDevice, static_cast<VkFormat>(format),
                                                     &props);

      if (tiling == vk::ImageTiling::eLinear
          && (vk::FormatFeatureFlags{props.linearTilingFeatures} & features) == features) {
        return format;
      } else if (tiling == vk::ImageTiling::eOptimal
                 && (vk::FormatFeatureFlags{props.optimalTilingFeatures} & features) == features) {
        return format;
      }
    }
    throw std::runtime_error(
        fmt::format("cannot find a suitable candidate (among {}) format that supports [{}] "
                    "tiling and has [{}] features",
                    candidates.size(), static_cast<std::underlying_type_t<vk::ImageTiling>>(tiling),
                    get_flag_value(features)));
  }

  vk::FormatProperties VulkanContext::getFormatProperties(vk::Format format) const noexcept {
    return physicalDevice.getFormatProperties(format, dispatcher);
  }

}  // namespace zs