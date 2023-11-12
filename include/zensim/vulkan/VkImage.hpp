#pragma once

#include "zensim/vulkan/VkBuffer.hpp"

namespace zs {

  struct VkTexture;
  struct Image {
    Image() = delete;
    Image(VulkanContext &ctx)
        : ctx{ctx},
          image{VK_NULL_HANDLE},
#if ZS_VULKAN_USE_VMA
          allocation{0},
#else
          pmem{},
#endif
          pview{} {
    }
    Image(Image &&o) noexcept
        : ctx{o.ctx},
          image{o.image},
#if ZS_VULKAN_USE_VMA
          allocation{o.allocation},
#else
          pmem{std::move(o.pmem)},
#endif
          pview{std::move(o.pview)} {
      o.pview = {};
      o.image = VK_NULL_HANDLE;
#if ZS_VULKAN_USE_VMA
      o.allocation = 0;
#else
      pmem.reset();
#endif
      o.pview.reset();
    }
    ~Image() {
      if (pview.has_value()) {
        ctx.device.destroyImageView(*pview, nullptr, ctx.dispatcher);
        pview.reset();
      }
      ctx.device.destroyImage(image, nullptr, ctx.dispatcher);
      image = VK_NULL_HANDLE;
#if ZS_VULKAN_USE_VMA
      vmaFreeMemory(ctx.allocator(), allocation);
#else
      pmem.reset();
#endif
    }

    vk::Image operator*() const { return image; }
    operator vk::Image() const { return image; }
    operator vk::ImageView() const { return *pview; }
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
    const VkMemory &memory() const { return *pmem; }
#endif
    bool hasView() const { return static_cast<bool>(pview); }
    const vk::ImageView &view() const { return *pview; }

  protected:
    friend struct VulkanContext;
    friend struct VkTexture;

    VulkanContext &ctx;
    vk::Image image;
#if ZS_VULKAN_USE_VMA
    VmaAllocation allocation;
#else
    std::shared_ptr<VkMemory> pmem;
#endif

    std::optional<vk::ImageView> pview;
  };

  struct ImageView {
    ImageView(VulkanContext &ctx) : ctx{ctx}, imgv{VK_NULL_HANDLE} {}
    ~ImageView() { ctx.device.destroyImageView(imgv, nullptr, ctx.dispatcher); }
    ImageView(ImageView &&o) noexcept : ctx{o.ctx}, imgv{o.imgv} { o.imgv = VK_NULL_HANDLE; }

    vk::ImageView operator*() const { return imgv; }
    operator vk::ImageView() const { return imgv; }

  protected:
    friend struct VulkanContext;

    VulkanContext &ctx;
    vk::ImageView imgv;
  };

  struct Framebuffer {
    Framebuffer(VulkanContext &ctx) : ctx{ctx}, framebuffer{VK_NULL_HANDLE} {}
    ~Framebuffer() { ctx.device.destroyFramebuffer(framebuffer, nullptr, ctx.dispatcher); }
    Framebuffer(Framebuffer &&o) noexcept : ctx{o.ctx}, framebuffer{o.framebuffer} {
      o.framebuffer = VK_NULL_HANDLE;
    }

    vk::Framebuffer operator*() const { return framebuffer; }
    operator vk::Framebuffer() const { return framebuffer; }

  protected:
    friend struct VulkanContext;

    VulkanContext &ctx;
    vk::Framebuffer framebuffer;
  };

}  // namespace zs