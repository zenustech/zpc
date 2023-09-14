#pragma once
#include <fstream>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace zs {

  inline std::vector<char> read_binary_file(std::string_view filePath) {
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("failed to open file" + filePath);
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
  }

  /// @ref legit engine
  inline bool is_depth_format(vk::Format format) noexcept {
    return format >= vk::Format::eD16Unorm && format < vk::Format::eD32SfloatS8Uint;
  }

  template <typename ET = float> inline vk::Format deduce_attribute_format(wrapt<ET> = {}) {
    if constexpr (is_arithmetic_v<ET>) {
      using T = ET;
      ///
      if constexpr (is_floating_point_v<T>) {
        // floating point scalar
        if constexpr (is_same_v<float>)
          return vk::Format::eR32Sfloat;
        else if constexpr (is_same_v<double>)
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
          if constexpr (is_same_v<float>)
            return vk::Format::eR32Sfloat;
          else if constexpr (is_same_v<double>)
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
          if constexpr (is_same_v<float>)
            return vk::Format::eR32G32Sfloat;
          else if constexpr (is_same_v<double>)
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
          if constexpr (is_same_v<float>)
            return vk::Format::eR32G32B32Sfloat;
          else if constexpr (is_same_v<double>)
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
          if constexpr (is_same_v<float>)
            return vk::Format::eR32G32B32A32Sfloat;
          else if constexpr (is_same_v<double>)
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