#pragma once
#include <optional>

#include "zensim/vulkan/VkContext.hpp"

namespace zs {

  struct BufferView;

  struct VkMemory {
    VkMemory() = delete;
    VkMemory(VulkanContext& ctx)
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
    friend struct VulkanContext;

    VulkanContext& ctx;
    vk::DeviceMemory mem;
    vk::DeviceSize memSize;
    vk::MemoryPropertyFlags memoryPropertyFlags;
  };

  struct Buffer {
    Buffer() = delete;
    Buffer(VulkanContext& ctx)
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
      o.pview = {};
      o.mapped = nullptr;
      o.usageFlags = {};
    }
    ~Buffer() {
      if (pview.has_value()) ctx.device.destroyBufferView(*pview, nullptr, ctx.dispatcher);
      ctx.device.destroyBuffer(buffer, nullptr, ctx.dispatcher);
    }

    /// access
    vk::Buffer operator*() const { return buffer; }
    operator vk::Buffer() const { return buffer; }
    const VkMemory& memory() const { return *pmem; }
    bool hasView() const { return static_cast<bool>(pview); }
    const vk::BufferView& view() const { return *pview; }

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

    /// @ref sascha willems Vulkan examples
    /**
     * Flush a memory range of the buffer to make it visible to the device
     *
     * @note Only required for non-coherent memory
     *
     * @param size (Optional) Size of the memory range to flush. Pass VK_WHOLE_SIZE to flush the
     * complete buffer range.
     * @param offset (Optional) Byte offset from beginning
     *
     * @return VkResult of the flush call
     */
    void flush(vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0) {
      vk::MappedMemoryRange mappedRange{};
      mappedRange.memory = *pmem;
      mappedRange.offset = offset;
      mappedRange.size = size;
      return ctx.device.flushMappedMemoryRanges({mappedRange}, ctx.dispatcher);
    }

    /**
     * Invalidate a memory range of the buffer to make it visible to the host
     *
     * @note Only required for non-coherent memory
     *
     * @param size (Optional) Size of the memory range to invalidate. Pass VK_WHOLE_SIZE to
     * invalidate the complete buffer range.
     * @param offset (Optional) Byte offset from beginning
     *
     * @return VkResult of the invalidate call
     */
    void invalidate(vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0) {
      vk::MappedMemoryRange mappedRange{};
      mappedRange.memory = *pmem;
      mappedRange.offset = offset;
      mappedRange.size = size;
      return ctx.device.invalidateMappedMemoryRanges({mappedRange}, ctx.dispatcher);
    }
#if 0
    // vk::DescriptorBufferInfo descriptor();

    vk::Result bind(vk::DeviceSize offset = 0);
    void setupDescriptor(vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0);
    void copyTo(void* data, vk::DeviceSize size);
    vk::Result flush(vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0);
    vk::Result invalidate(vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0);
#endif

  protected:
    friend struct VulkanContext;

    VulkanContext& ctx;
    vk::Buffer buffer;
    vk::DeviceSize size, alignment;
    std::shared_ptr<VkMemory> pmem;

    std::optional<vk::BufferView> pview;
    void* mapped;

    vk::BufferUsageFlags usageFlags;
  };

  struct BufferView {
    BufferView(VulkanContext& ctx) : ctx{ctx}, bufv{} {}
    BufferView(const BufferView&) = delete;
    BufferView(BufferView&&) noexcept = default;
    ~BufferView() { ctx.device.destroyBufferView(bufv, nullptr, ctx.dispatcher); }
    vk::BufferView operator*() const { return bufv; }
    operator vk::BufferView() const { return bufv; }

  protected:
    VulkanContext& ctx;
    vk::BufferView bufv;
  };

}  // namespace zs