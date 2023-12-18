//
#define THSVS_SIMPLER_VULKAN_SYNCHRONIZATION_IMPLEMENTATION
#include "zensim/vulkan/VkUtils.hpp"

#include "zensim/zpc_tpls/magic_enum/magic_enum.hpp"

namespace zs {

  std::vector<char> read_binary_file(std::string_view filePath) {
    std::ifstream file(filePath.data(), std::ios::ate | std::ios::binary | std::ios::in);
    if (!file.is_open())
      throw std::runtime_error(std::string("failed to open file") + filePath.data());
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
  }

  vk::ImageMemoryBarrier image_layout_transition_barrier(vk::Image image,
                                                         vk::ImageLayout oldImageLayout,
                                                         vk::ImageLayout newImageLayout,
                                                         vk::ImageSubresourceRange subresourceRange,
                                                         vk::PipelineStageFlags srcStageMask,
                                                         vk::PipelineStageFlags dstStageMask) {
    auto imageMemoryBarrier = vk::ImageMemoryBarrier{}
                                  .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                                  .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    imageMemoryBarrier.oldLayout = oldImageLayout;
    imageMemoryBarrier.newLayout = newImageLayout;
    imageMemoryBarrier.image = image;
    imageMemoryBarrier.subresourceRange = subresourceRange;

    // Source layouts (old)
    // Source access mask controls actions that have to be finished on the old
    // layout before it will be transitioned to the new layout
    switch (oldImageLayout) {
      case vk::ImageLayout::eUndefined:
        // Image layout is undefined (or does not matter)
        // Only valid as initial layout
        // No flags required, listed only for completeness
        imageMemoryBarrier.srcAccessMask = {};
        break;

      case vk::ImageLayout::ePreinitialized:
        // Image is preinitialized
        // Only valid as initial layout for linear images, preserves memory
        // contents Make sure host writes have been finished
        imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eHostWrite;
        break;

      case vk::ImageLayout::eColorAttachmentOptimal:
        // Image is a color attachment
        // Make sure any writes to the color buffer have been finished
        imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
        break;

      case vk::ImageLayout::eDepthStencilAttachmentOptimal:
        // Image is a depth/stencil attachment
        // Make sure any writes to the depth/stencil buffer have been finished
        imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        break;

      case vk::ImageLayout::eTransferSrcOptimal:
        // Image is a transfer source
        // Make sure any reads from the image have been finished
        imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        break;

      case vk::ImageLayout::eTransferDstOptimal:
        // Image is a transfer destination
        // Make sure any writes to the image have been finished
        imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        break;

      case vk::ImageLayout::eShaderReadOnlyOptimal:
        // Image is read by a shader
        // Make sure any shader reads from the image have been finished
        imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
        break;
      default:
        // Other source layouts aren't handled (yet)
        break;
    }

    // Target layouts (new)
    // Destination access mask controls the dependency for the new image
    // layout
    switch (newImageLayout) {
      case vk::ImageLayout::eTransferDstOptimal:
        // Image will be used as a transfer destination
        // Make sure any writes to the image have been finished
        imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        break;

      case vk::ImageLayout::eTransferSrcOptimal:
        // Image will be used as a transfer source
        // Make sure any reads from the image have been finished
        imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
        break;

      case vk::ImageLayout::eColorAttachmentOptimal:
        // Image will be used as a color attachment
        // Make sure any writes to the color buffer have been finished
        imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
        break;

      case vk::ImageLayout::eDepthStencilAttachmentOptimal:
        // Image layout will be used as a depth/stencil attachment
        // Make sure any writes to depth/stencil buffer have been finished
        imageMemoryBarrier.dstAccessMask
            = imageMemoryBarrier.dstAccessMask | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        break;

      case vk::ImageLayout::eShaderReadOnlyOptimal:
        // Image will be read in a shader (sampler, input attachment)
        // Make sure any writes to the image have been finished
        if (imageMemoryBarrier.srcAccessMask == vk::AccessFlags{}) {
          imageMemoryBarrier.srcAccessMask
              = vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eTransferWrite;
        }
        imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        break;
      default:
        // Other source layouts aren't handled (yet)
        break;
    }
    return imageMemoryBarrier;
  }

  vk::ImageMemoryBarrier image_layout_transition_barrier(vk::Image image,
                                                         vk::ImageAspectFlags aspectMask,
                                                         vk::ImageLayout oldImageLayout,
                                                         vk::ImageLayout newImageLayout,
                                                         vk::PipelineStageFlags srcStageMask,
                                                         vk::PipelineStageFlags dstStageMask) {
    auto subresourceRange = vk::ImageSubresourceRange{}
                                .setAspectMask(aspectMask)
                                .setBaseMipLevel(0)
                                .setLevelCount(1)
                                .setLayerCount(1);
    return image_layout_transition_barrier(image, oldImageLayout, newImageLayout, subresourceRange,
                                           srcStageMask, dstStageMask);
  }

  template <typename VkEnumT> std::string reflect_vk_enum(VkEnumT e) {
    return std::string(magic_enum::enum_name(e));
  }

  ZPC_INSTANTIATE std::string reflect_vk_enum<vk::Format>(vk::Format);
  ZPC_INSTANTIATE std::string reflect_vk_enum<vk::SampleCountFlagBits>(vk::SampleCountFlagBits);
  ZPC_INSTANTIATE std::string reflect_vk_enum<vk::ShaderStageFlagBits>(vk::ShaderStageFlagBits);
  ZPC_INSTANTIATE std::string reflect_vk_enum<vk::DescriptorType>(vk::DescriptorType);

  bool is_color_format(vk::Format format) noexcept {
    auto name = magic_enum::enum_name(format);
    return name[1] == 'R' || name[1] == 'G' || name[1] == 'B' || name[1] == 'A' || name[1] == 'E';
  }
  vk::ImageAspectFlags deduce_image_format_aspect_flag(vk::Format format) noexcept {
    vk::ImageAspectFlags flag{};
    if (is_depth_stencil_format(format)) {
      flag |= vk::ImageAspectFlagBits::eDepth;
      if (format == vk::Format::eS8Uint || format == vk::Format::eD16UnormS8Uint
          || format == vk::Format::eD24UnormS8Uint || format == vk::Format::eD32SfloatS8Uint)
        flag |= vk::ImageAspectFlagBits::eStencil;
    } else {
      flag |= vk::ImageAspectFlagBits::eColor;
    }
    return flag;
  }

}  // namespace zs