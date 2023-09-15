#pragma once

#include "zensim/vulkan/VkBuffer.hpp"

namespace zs {

  struct Image {
    Image() = delete;
    Image(Vulkan::VulkanContext &ctx) : ctx{ctx}, image{VK_NULL_HANDLE}, pmem{}, pview{} {}
    Image(Image &&o) noexcept
        : ctx{o.ctx}, image{o.image}, pmem{std::move(o.pmem)}, pview{std::move(o.pview)} {
      o.pview = {};
      o.image = VK_NULL_HANDLE;
    }
    ~Image() {
      if (pview.has_value()) ctx.device.destroyImageView(*pview, nullptr, ctx.dispatcher);
      ctx.device.destroyImage(image, nullptr, ctx.dispatcher);
    }

    vk::Image operator*() const { return image; }
    operator vk::Image() const { return image; }
    const VkMemory &memory() const { return *pmem; }
    bool hasView() const { return static_cast<bool>(pview); }
    const vk::ImageView &view() const { return *pview; }

  protected:
    friend struct Vulkan::VulkanContext;

    Vulkan::VulkanContext &ctx;
    vk::Image image;
    std::shared_ptr<VkMemory> pmem;

    std::optional<vk::ImageView> pview;
  };

  struct ImageView {
    ImageView(Vulkan::VulkanContext &ctx) : ctx{ctx}, imgv{VK_NULL_HANDLE} {}
    ~ImageView() { ctx.device.destroyImageView(imgv, nullptr, ctx.dispatcher); }
    ImageView(ImageView &&o) noexcept : ctx{o.ctx}, imgv{o.imgv} { o.imgv = VK_NULL_HANDLE; }

    vk::ImageView operator*() const { return imgv; }
    operator vk::ImageView() const { return imgv; }

  protected:
    friend struct Vulkan::VulkanContext;

    Vulkan::VulkanContext &ctx;
    vk::ImageView imgv;
  };

  struct Framebuffer {
    Framebuffer(Vulkan::VulkanContext &ctx) : ctx{ctx}, framebuffer{VK_NULL_HANDLE} {}
    ~Framebuffer() { ctx.device.destroyFramebuffer(framebuffer, nullptr, ctx.dispatcher); }
    Framebuffer(Framebuffer &&o) noexcept : ctx{o.ctx}, framebuffer{o.framebuffer} {
      o.framebuffer = VK_NULL_HANDLE;
    }

    vk::Framebuffer operator*() const { return framebuffer; }
    operator vk::Framebuffer() const { return framebuffer; }

  protected:
    friend struct Vulkan::VulkanContext;

    Vulkan::VulkanContext &ctx;
    vk::Framebuffer framebuffer;
  };

}  // namespace zs