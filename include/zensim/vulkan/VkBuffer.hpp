#pragma once
#include "zensim/vulkan/Vulkan.hpp"

namespace zs {

  struct Vulkan;

  struct VkMemory {
    VkMemory() = default;
    VkMemory(Vulkan::VulkanContext& ctx)
        : ctx{ctx}, mem{VK_NULL_HANDLE}, memSize{0}, memoryPropertyFlags{} {}
    VkMemory(VkMemory&& o) noexcept
        : ctx{o.ctx}, mem{o.mem}, memSize{o.memSize}, memoryPropertyFlags{o.memoryPropertyFlags} {
      o.mem = VK_NULL_HANDLE;
      o.memSize = 0;
      o.memoryPropertyFlags = {};
    }
    ~VkMemory() { ctx.device.freeMemory(mem, nullptr, ctx.dispatcher); }

    vk::DeviceSize size() const { return memSize; }

    vk::DeviceMemory operator*() const { return mem; }
    operator vk::DeviceMemory() const { return mem; }

  protected:
    friend struct Vulkan::VulkanContext;

    Vulkan::VulkanContext& ctx;
    vk::DeviceMemory mem;
    vk::DeviceSize memSize;
    vk::MemoryPropertyFlags memoryPropertyFlags;
  };

  struct Buffer {
    Buffer() = delete;
    Buffer(Vulkan::VulkanContext& ctx)
        : ctx{ctx},
          buffer{VK_NULL_HANDLE},
          size{0},
          alignment{0},
          pmem{},
          pview{},
          mapped{nullptr},
          usageFlags{} {}
    Buffer(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept
        : ctx{o.ctx},
          buffer{o.buffer},
          size{o.size},
          alignment{o.alignment},
          pmem{std::move(o.pmem)},
          pview{std::move(o.pview)},
          mapped{o.mapped},
          usageFlags{o.usageFlags} {
      o.buffer = VK_NULL_HANDLE;
      o.size = 0;
      o.alignment = 0;
      o.mapped = nullptr;
      o.usageFlags = {};
    }
    ~Buffer() { ctx.device.destroyBuffer(buffer, nullptr, ctx.dispatcher); }

    struct BufferView {
      BufferView(Vulkan::VulkanContext& ctx) : ctx{ctx}, bufv{} {}
      BufferView(const BufferView&) = delete;
      BufferView(BufferView&&) noexcept = default;
      ~BufferView() { ctx.device.destroyBufferView(bufv, nullptr, ctx.dispatcher); }
      vk::BufferView operator*() const { return bufv; }
      operator vk::BufferView() const { return bufv; }

    protected:
      Vulkan::VulkanContext& ctx;
      vk::BufferView bufv;
    };

    /// access
    vk::Buffer operator*() const { return buffer; }
    operator vk::Buffer() const { return buffer; }
    const VkMemory& memory() const { return *pmem; }
    bool hasView() const { return static_cast<bool>(pview); }
    const BufferView& view() const { return *pview; }

#if 0
    // vk::DescriptorBufferInfo descriptor();

    vk::Result map(vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0);
    void unmap();
    vk::Result bind(vk::DeviceSize offset = 0);
    void setupDescriptor(vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0);
    void copyTo(void* data, vk::DeviceSize size);
    vk::Result flush(vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0);
    vk::Result invalidate(vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0);
#endif

  protected:
    friend struct Vulkan::VulkanContext;

    Vulkan::VulkanContext& ctx;
    vk::Buffer buffer;
    vk::DeviceSize size;
    vk::DeviceSize alignment;
    std::shared_ptr<VkMemory> pmem;

    std::unique_ptr<BufferView> pview;
    void* mapped;

    vk::BufferUsageFlags usageFlags;
  };

  template <typename ET = float> vk::Format deduce_attribute_format(wrapt<ET> = {}) {
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