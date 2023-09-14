#pragma once
#include "zensim/vulkan/Vulkan.hpp"

namespace zs {

  struct Vulkan;

  struct VkMemory {
    VkMemory() = delete;
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

    void map(vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0) {
      mapped = ctx.device.mapMemory(memory(), offset, size, vk::MemoryMapFlags{}, ctx.dispatcher);
    }
    void unmap() {
      if (mapped) {
        ctx.device.unmapMemory(memory(), ctx.dispatcher);
        mapped = nullptr;
      };
    }
    void* mappedAddress() const { return mapped; }
#if 0
    // vk::DescriptorBufferInfo descriptor();

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
    vk::DeviceSize size, alignment;
    std::shared_ptr<VkMemory> pmem;

    std::unique_ptr<BufferView> pview;
    void* mapped;

    vk::BufferUsageFlags usageFlags;
  };

}  // namespace zs