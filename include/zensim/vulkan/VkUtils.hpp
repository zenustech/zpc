#pragma once
#include <cmath>
#include <fstream>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "zensim/ZpcMeta.hpp"

namespace zs {

  /// @ref little vulkan engine
  inline std::vector<char> read_binary_file(std::string_view filePath) {
    std::ifstream file(filePath.data(), std::ios::ate | std::ios::binary);
    if (!file.is_open())
      throw std::runtime_error(std::string("failed to open file") + filePath.data());
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
  }

  template <typename BitType> constexpr auto get_flag_value(vk::Flags<BitType> flags) {
    // using MaskType = typename vk::Flags<BitType>::MaskType;
    using MaskType = typename std::underlying_type_t<BitType>;
    return static_cast<MaskType>(flags);
  }

  constexpr vk::DeviceSize get_aligned_size(vk::DeviceSize size, vk::DeviceSize alignment) {
    /// @note both size and alignment are in bytes
    if (alignment > 0) return (size + alignment - 1) & ~(alignment - 1);
    return size;
  }

  inline u32 get_num_mip_levels(const vk::Extent2D &extent) {
    return static_cast<u32>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1;
  }

  inline u32 get_num_mip_levels(const vk::Extent3D &extent) {
    return static_cast<u32>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1;
  }

  /// @ref legit engine
  constexpr bool is_depth_format(vk::Format format) noexcept {
    return format >= vk::Format::eD16Unorm && format < vk::Format::eD32SfloatS8Uint;
  }
  constexpr vk::ImageUsageFlags get_general_usage_flags(vk::Format format) {
    vk::ImageUsageFlags usageFlags = vk::ImageUsageFlagBits::eSampled;
    if (is_depth_format(format)) {
      usageFlags
          |= vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled;
    } else {
      usageFlags |= vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst
                    | vk::ImageUsageFlagBits::eSampled;
    }
    return usageFlags;
  }
  constexpr vk::ImageUsageFlags g_colorImageUsage = vk::ImageUsageFlagBits::eColorAttachment
                                                    | vk::ImageUsageFlagBits::eTransferDst
                                                    | vk::ImageUsageFlagBits::eSampled;
  constexpr vk::ImageUsageFlags g_depthImageUsage
      = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled;

#if 0
  /// @ref nvpro core
  inline vk::PipelineStageFlags make_access_mask_pipeline_stage_flags(
      vk::AccessFlags accessMask, vk::PipelineStageFlags supportedShaderBits) {
    static constexpr vk::AccessFlagBits accessBits[]
        = { vk::AccessFlagBits::eIndirectCommandRead,
            vk::AccessFlagBits::eIndexRead,
            vk::AccessFlagBits::eVertexAttributeRead,
            vk::AccessFlagBits::eUniformRead,
            vk::AccessFlagBits::eInputAttachmentRead,

            vk::AccessFlagBits::eShaderRead,
            vk::AccessFlagBits::eShaderWrite vk::AccessFlagBits:: :
                eColorAttachmentRead;  // VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
  };
  vk::AccessFlagBits::VK_ACCESS_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT,
      vk::AccessFlagBits::VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,

      vk::AccessFlagBits::VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
      vk::AccessFlagBits::VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      vk::AccessFlagBits::VK_ACCESS_TRANSFER_READ_BIT,
      vk::AccessFlagBits::VK_ACCESS_TRANSFER_WRITE_BIT, vk::AccessFlagBits::VK_ACCESS_HOST_READ_BIT,

      vk::AccessFlagBits::VK_ACCESS_HOST_WRITE_BIT, vk::AccessFlagBits::VK_ACCESS_MEMORY_READ_BIT,
      vk::AccessFlagBits::VK_ACCESS_MEMORY_WRITE_BIT,
};
static constexpr vk::PipelineStageFlags pipeStages[]
    = {vk::PipelineStageFlagBits::eDrawIndirect,
       vk::PipelineStageFlagBits::eVertexInput,
       vk::PipelineStageFlagBits::eVertexInput,
       supportedShaderBits,
       vk::PipelineStageFlagBits::eFragmentShader,

       supportedShaderBits,
       supportedShaderBits,
       vk::PipelineStageFlagBits::eColorAttachmentOutput,
       vk::PipelineStageFlagBits::eColorAttachmentOutput,
       vk::PipelineStageFlagBits::eColorAttachmentOutput,

       vk::PipelineStageFlagBits::VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
           | vk::PipelineStageFlagBits::VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
       vk::PipelineStageFlagBits::VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
           | vk::PipelineStageFlagBits::VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
       vk::PipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT,
       vk::PipelineStageFlagBits::VK_PIPELINE_STAGE_TRANSFER_BIT,
       vk::PipelineStageFlagBits::VK_PIPELINE_STAGE_HOST_BIT,

       vk::PipelineStageFlagBits::VK_PIPELINE_STAGE_HOST_BIT,
       {},
       {}};
if (!accessMask) return vk::PipelineStageFlagBits::eTopOfPipe;

vk::PipelineStageFlags pipes = 0;

for (u32 i = 0; i < sizeof(accessBits); i++) {
  if (accessBits[i] & accessMask) {
    pipes |= pipeStages[i];
  }
}
if (!pipes)
  throw std::runtime_error(
      fmt::format("none of the pipeline stages compatible with the access mask [{}]",
                  static_cast<typename vk::AccessFlags::MaskType>(accessMask)));

return pipes;
}

#endif
  template <typename ET = float> constexpr vk::Format deduce_attribute_format(wrapt<ET> = {}) {
    if constexpr (is_arithmetic_v<ET>) {
      using T = ET;
      ///
      if constexpr (is_floating_point_v<T>) {
        // floating point scalar
        if constexpr (is_same_v<T, float>)
          return vk::Format::eR32Sfloat;
        else if constexpr (is_same_v<T, double>)
          return vk::Format::eR64Sfloat;
        else
          static_assert(always_false<T>, "unable to deduce this attribute format!\n");
      } else if constexpr (is_integral_v<T>) {
        // integral scalar
        if constexpr (is_signed_v<T>) {
          // signed
          if constexpr (sizeof(T) == 1)
            return vk::Format::eR8Sint;
          else if constexpr (sizeof(T) == 2)
            return vk::Format::eR16Sint;
          else if constexpr (sizeof(T) == 4)
            return vk::Format::eR32Sint;
          else if constexpr (sizeof(T) == 8)
            return vk::Format::eR64Sint;
          else
            static_assert(always_false<T>, "unable to deduce this attribute format!\n");
        } else {
          // unsigned
          if constexpr (sizeof(T) == 1)
            return vk::Format::eR8Uint;
          else if constexpr (sizeof(T) == 2)
            return vk::Format::eR16Uint;
          else if constexpr (sizeof(T) == 4)
            return vk::Format::eR32Uint;
          else if constexpr (sizeof(T) == 8)
            return vk::Format::eR64Uint;
          else
            static_assert(always_false<T>, "unable to deduce this attribute format!\n");
        }
      }
      ///
    } else {
      using T = typename ET::value_type;
      constexpr size_t N = sizeof(ET) / sizeof(T);
      if constexpr (N == 1) {
        /// N = 1
        if constexpr (is_floating_point_v<T>) {
          // floating point scalar
          if constexpr (is_same_v<T, float>)
            return vk::Format::eR32Sfloat;
          else if constexpr (is_same_v<T, double>)
            return vk::Format::eR64Sfloat;
          else
            static_assert(always_false<T>, "unable to deduce this attribute format!\n");
        } else if constexpr (is_integral_v<T>) {
          // integral scalar
          if constexpr (is_signed_v<T>) {
            // signed
            if constexpr (sizeof(T) == 1)
              return vk::Format::eR8Sint;
            else if constexpr (sizeof(T) == 2)
              return vk::Format::eR16Sint;
            else if constexpr (sizeof(T) == 4)
              return vk::Format::eR32Sint;
            else if constexpr (sizeof(T) == 8)
              return vk::Format::eR64Sint;
            else
              static_assert(always_false<T>, "unable to deduce this attribute format!\n");
          } else {
            // unsigned
            if constexpr (sizeof(T) == 1)
              return vk::Format::eR8Uint;
            else if constexpr (sizeof(T) == 2)
              return vk::Format::eR16Uint;
            else if constexpr (sizeof(T) == 4)
              return vk::Format::eR32Uint;
            else if constexpr (sizeof(T) == 8)
              return vk::Format::eR64Uint;
            else
              static_assert(always_false<T>, "unable to deduce this attribute format!\n");
          }
        }
        /// N = 1
      } else if constexpr (N == 2) {
        /// N = 2
        if constexpr (is_floating_point_v<T>) {
          // floating point scalar
          if constexpr (is_same_v<T, float>)
            return vk::Format::eR32G32Sfloat;
          else if constexpr (is_same_v<T, double>)
            return vk::Format::eR64G64Sfloat;
          else
            static_assert(always_false<T>, "unable to deduce this attribute format!\n");
        } else if constexpr (is_integral_v<T>) {
          // integral scalar
          if constexpr (is_signed_v<T>) {
            // signed
            if constexpr (sizeof(T) == 1)
              return vk::Format::eR8G8Sint;
            else if constexpr (sizeof(T) == 2)
              return vk::Format::eR16G16Sint;
            else if constexpr (sizeof(T) == 4)
              return vk::Format::eR32G32Sint;
            else if constexpr (sizeof(T) == 8)
              return vk::Format::eR64G64Sint;
            else
              static_assert(always_false<T>, "unable to deduce this attribute format!\n");
          } else {
            // unsigned
            if constexpr (sizeof(T) == 1)
              return vk::Format::eR8G8Uint;
            else if constexpr (sizeof(T) == 2)
              return vk::Format::eR16G16Uint;
            else if constexpr (sizeof(T) == 4)
              return vk::Format::eR32G32Uint;
            else if constexpr (sizeof(T) == 8)
              return vk::Format::eR64G64Uint;
            else
              static_assert(always_false<T>, "unable to deduce this attribute format!\n");
          }
        }
        /// N = 2
      } else if constexpr (N == 3) {
        /// N = 3
        if constexpr (is_floating_point_v<T>) {
          // floating point scalar
          if constexpr (is_same_v<T, float>)
            return vk::Format::eR32G32B32Sfloat;
          else if constexpr (is_same_v<T, double>)
            return vk::Format::eR64G64B64Sfloat;
          else
            static_assert(always_false<T>, "unable to deduce this attribute format!\n");
        } else if constexpr (is_integral_v<T>) {
          // integral scalar
          if constexpr (is_signed_v<T>) {
            // signed
            if constexpr (sizeof(T) == 1)
              return vk::Format::eR8G8B8Sint;
            else if constexpr (sizeof(T) == 2)
              return vk::Format::eR16G16B16Sint;
            else if constexpr (sizeof(T) == 4)
              return vk::Format::eR32G32B32Sint;
            else if constexpr (sizeof(T) == 8)
              return vk::Format::eR64G64B64Sint;
            else
              static_assert(always_false<T>, "unable to deduce this attribute format!\n");
          } else {
            // unsigned
            if constexpr (sizeof(T) == 1)
              return vk::Format::eR8G8B8Uint;
            else if constexpr (sizeof(T) == 2)
              return vk::Format::eR16G16B16Uint;
            else if constexpr (sizeof(T) == 4)
              return vk::Format::eR32G32B32Uint;
            else if constexpr (sizeof(T) == 8)
              return vk::Format::eR64G64B64Uint;
            else
              static_assert(always_false<T>, "unable to deduce this attribute format!\n");
          }
        }
        /// N = 3
      } else if constexpr (N == 4) {
        /// N = 4
        if constexpr (is_floating_point_v<T>) {
          // floating point scalar
          if constexpr (is_same_v<T, float>)
            return vk::Format::eR32G32B32A32Sfloat;
          else if constexpr (is_same_v<T, double>)
            return vk::Format::eR64G64B64A64Sfloat;
          else
            static_assert(always_false<T>, "unable to deduce this attribute format!\n");
        } else if constexpr (is_integral_v<T>) {
          // integral scalar
          if constexpr (is_signed_v<T>) {
            // signed
            if constexpr (sizeof(T) == 1)
              return vk::Format::eR8G8B8A8Sint;
            else if constexpr (sizeof(T) == 2)
              return vk::Format::eR16G16B16A16Sint;
            else if constexpr (sizeof(T) == 4)
              return vk::Format::eR32G32B32A32Sint;
            else if constexpr (sizeof(T) == 8)
              return vk::Format::eR64G64B64A64Sint;
            else
              static_assert(always_false<T>, "unable to deduce this attribute format!\n");
          } else {
            // unsigned
            if constexpr (sizeof(T) == 1)
              return vk::Format::eR8G8B8A8Uint;
            else if constexpr (sizeof(T) == 2)
              return vk::Format::eR16G16B16A16Uint;
            else if constexpr (sizeof(T) == 4)
              return vk::Format::eR32G32B32A32Uint;
            else if constexpr (sizeof(T) == 8)
              return vk::Format::eR64G64B64A64Uint;
            else
              static_assert(always_false<T>, "unable to deduce this attribute format!\n");
          }
        }
        /// N = 4
      } else
        static_assert(always_false<T>, "unable to deduce this attribute format!\n");
    }
    return vk::Format::eUndefined;
  }

}  // namespace zs