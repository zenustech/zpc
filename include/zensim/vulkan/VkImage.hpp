#pragma once

#include "zensim/vulkan/VkBuffer.hpp"

namespace zs {

  struct VkTexture;
  struct ZPC_CORE_API Image {
    Image() = delete;
    Image(VulkanContext &ctx, vk::Image img = VK_NULL_HANDLE)
        : ctx{ctx},
          image{img},
#if ZS_VULKAN_USE_VMA
          allocation{0},
#else
          pmem{},
#endif
          pview{},
          usage{},
          extent{},
          mipLevels{1} {
    }
    Image(Image &&o) noexcept
        : ctx{o.ctx},
          image{o.image},
#if ZS_VULKAN_USE_VMA
          allocation{o.allocation},
#else
          pmem{std::move(o.pmem)},
#endif
          pview{std::move(o.pview)},
          usage{o.usage},
          extent{o.extent},
          mipLevels{o.mipLevels} {
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

    vk::DeviceSize getSize() const noexcept {
#if ZS_VULKAN_USE_VMA
      VmaAllocationInfo allocInfo;
      vmaGetAllocationInfo(ctx.allocator(), allocation, &allocInfo);
      return allocInfo.size;
#else
      return memory().memSize;
#endif
    }
    vk::Extent3D getExtent() const noexcept { return extent; }

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
    vk::ImageUsageFlags usage;
    vk::Extent3D extent;
    u32 mipLevels;
  };

  struct ImageView {
    ImageView(VulkanContext &ctx, vk::ImageView imgv = VK_NULL_HANDLE) : ctx{ctx}, imgv{imgv} {}
    ~ImageView() { ctx.device.destroyImageView(imgv, nullptr, ctx.dispatcher); }
    ImageView(ImageView &&o) noexcept : ctx{o.ctx}, imgv{o.imgv} { o.imgv = VK_NULL_HANDLE; }

    vk::ImageView operator*() const { return imgv; }
    operator vk::ImageView() const { return imgv; }

  protected:
    friend struct VulkanContext;

    VulkanContext &ctx;
    vk::ImageView imgv;
  };

  struct ImageSampler {
    ImageSampler(VulkanContext &ctx, vk::Sampler sampler = VK_NULL_HANDLE)
        : ctx{ctx}, sampler{sampler} {}
    ~ImageSampler() { ctx.device.destroySampler(sampler, nullptr, ctx.dispatcher); }
    ImageSampler(ImageSampler &&o) noexcept : ctx{o.ctx}, sampler{o.sampler} {
      o.sampler = VK_NULL_HANDLE;
    }

    vk::Sampler operator*() const { return sampler; }
    operator vk::Sampler() const { return sampler; }

  protected:
    friend struct VulkanContext;

    VulkanContext &ctx;
    vk::Sampler sampler;
  };

  struct Framebuffer {
    Framebuffer(VulkanContext &ctx, vk::Framebuffer fb = VK_NULL_HANDLE)
        : ctx{ctx}, framebuffer{fb} {}
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