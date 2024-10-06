#pragma once
#include <optional>

#include "zensim/vulkan/VkContext.hpp"

namespace zs {

  inline VmaMemoryUsage vk_to_vma_memory_usage(vk::MemoryPropertyFlags flags) {
    if ((flags & vk::MemoryPropertyFlagBits::eDeviceLocal)
        == vk::MemoryPropertyFlagBits::eDeviceLocal)
      return VMA_MEMORY_USAGE_GPU_ONLY;
    else if ((flags & vk::MemoryPropertyFlagBits::eHostCoherent)
             == vk::MemoryPropertyFlagBits::eHostCoherent)
      return VMA_MEMORY_USAGE_CPU_ONLY;
    else if ((flags & vk::MemoryPropertyFlagBits::eHostVisible)
             == vk::MemoryPropertyFlagBits::eHostVisible)
      return VMA_MEMORY_USAGE_CPU_TO_GPU;
    return VMA_MEMORY_USAGE_UNKNOWN;
  }

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

    vk::DeviceMemory operator*() const { return mem; }
    operator vk::DeviceMemory() const { return mem; }

  protected:
    friend struct VulkanContext;

    VulkanContext& ctx;
    vk::DeviceMemory mem;
    vk::DeviceSize memSize;
    vk::MemoryPropertyFlags memoryPropertyFlags;
  };

  struct VkMemoryRange {
    vk::DeviceMemory operator*() const { return memory; }
    operator vk::DeviceMemory() const { return memory; }

    operator vk::MappedMemoryRange() const {
      vk::MappedMemoryRange mappedRange{};
      mappedRange.memory = memory;
      mappedRange.offset = offset;
      mappedRange.size = size;
      return mappedRange;
    }

    vk::DeviceMemory memory;
    vk::DeviceSize offset;
    vk::DeviceSize size;
  };

  struct ZPC_CORE_API Buffer {
    Buffer() = delete;
    Buffer(VulkanContext& ctx)
        : ctx{ctx},
          buffer{VK_NULL_HANDLE},
          size{0},
          alignment{0},
#if ZS_VULKAN_USE_VMA
          allocation{0},
#else
          pmem{},
#endif
          pview{},
          mapped{nullptr},
          usageFlags{} {
    }
    Buffer(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept
        : ctx{o.ctx},
          buffer{o.buffer},
          size{o.size},
          alignment{o.alignment},
#if ZS_VULKAN_USE_VMA
          allocation{o.allocation},
#else
          pmem{std::move(o.pmem)},
#endif
          pview{std::move(o.pview)},
          mapped{o.mapped},
          usageFlags{o.usageFlags} {
      o.buffer = VK_NULL_HANDLE;
      o.size = 0;
      o.alignment = 0;
#if ZS_VULKAN_USE_VMA
      o.allocation = 0;
#endif
      o.pview = {};
      o.mapped = nullptr;
      o.usageFlags = {};
    }
    ~Buffer() {
      if (pview.has_value()) ctx.device.destroyBufferView(*pview, nullptr, ctx.dispatcher);
      unmap();
      ctx.device.destroyBuffer(buffer, nullptr, ctx.dispatcher);
#if ZS_VULKAN_USE_VMA
      vmaFreeMemory(ctx.allocator(), allocation);
#endif
    }
    void moveAssign(Buffer&& o) {
      if (&ctx != &o.ctx) throw std::runtime_error("unable to swap vk buffers due to ctx mismatch");
      std::swap(buffer, o.buffer);
      std::swap(size, o.size);
      std::swap(alignment, o.alignment);
#if ZS_VULKAN_USE_VMA
      std::swap(allocation, o.allocation);
#else
      std::swap(pmem, o.pmem);
#endif
      std::swap(pview, o.pview);
      std::swap(mapped, o.mapped);
      std::swap(usageFlags, o.usageFlags);
    }

    /// access
    vk::Buffer operator*() const { return buffer; }
    operator vk::Buffer() const { return buffer; }
#if ZS_VULKAN_USE_VMA
    VkMemoryRange memory() const {
      VmaAllocationInfo allocInfo;
      vmaGetAllocationInfo(ctx.allocator(), allocation, &allocInfo);

      VkMemoryRange memRange;
      memRange.memory = allocInfo.deviceMemory;
      memRange.offset = allocInfo.offset;
      memRange.size = allocInfo.size;
      return memRange;
    }
#else
    const VkMemory& memory() const { return *pmem; }
#endif
    bool hasView() const { return static_cast<bool>(pview); }
    const vk::BufferView& view() const { return *pview; }

#if ZS_VULKAN_USE_VMA
    void map() {
      VkResult result = vmaMapMemory(ctx.allocator(), allocation, &mapped);
      if (result != VK_SUCCESS) throw std::runtime_error("unable to map this buffer memory");
    }
#else
    void map(vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0) {
      mapped = ctx.device.mapMemory(memory(), offset, size, vk::MemoryMapFlags{}, ctx.dispatcher);
    }
#endif
    void unmap() {
      if (mapped) {
#if ZS_VULKAN_USE_VMA
        vmaUnmapMemory(ctx.allocator(), allocation);
#else
        ctx.device.unmapMemory(memory(), ctx.dispatcher);
#endif
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
#if ZS_VULKAN_USE_VMA
    void flush() {
      auto mappedRange = memory();
      vmaFlushAllocation(ctx.allocator(), allocation, mappedRange.offset, mappedRange.size);
    }
#else
    void flush(vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0) {
      vk::MappedMemoryRange mappedRange{};
      mappedRange.memory = *pmem;
      mappedRange.offset = offset;
      mappedRange.size = size;
      return ctx.device.flushMappedMemoryRanges({mappedRange}, ctx.dispatcher);
    }
#endif

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
#if ZS_VULKAN_USE_VMA
    void invalidate() {
      auto mappedRange = memory();
      vmaInvalidateAllocation(ctx.allocator(), allocation, mappedRange.offset, mappedRange.size);
    }
#else
    void invalidate(vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0) {
      vk::MappedMemoryRange mappedRange{};
      mappedRange.memory = *pmem;
      mappedRange.offset = offset;
      mappedRange.size = size;
      return ctx.device.invalidateMappedMemoryRanges({mappedRange}, ctx.dispatcher);
    }
#endif

    vk::DeviceSize getSize() const noexcept { return size; }
    vk::DeviceSize getAlignment() const noexcept { return alignment; }
    vk::DescriptorBufferInfo descriptorInfo() {
      return vk::DescriptorBufferInfo{buffer, (u32)0, size};
    }
    VulkanContext *pCtx() noexcept { return &ctx; }
    const VulkanContext *pCtx() const noexcept { return &ctx; }

  protected:
    friend struct VulkanContext;
    friend struct VkModel;

    VulkanContext& ctx;
    vk::Buffer buffer;
    vk::DeviceSize size, alignment;

#if ZS_VULKAN_USE_VMA
    VmaAllocation allocation;
#else
    std::shared_ptr<VkMemory> pmem;
#endif

    std::optional<vk::BufferView> pview;
    void* mapped;

    vk::BufferUsageFlags usageFlags;
  };

  struct ZPC_CORE_API BufferView {
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