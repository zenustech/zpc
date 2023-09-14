#pragma once
#include "zensim/vulkan/VkBuffer.hpp"

namespace zs {

  struct Image {
    Image(Vulkan::VulkanContext &ctx) : ctx{ctx}, image{VK_NULL_HANDLE} {}
    ~Image() { ctx.device.destroyImage(image, nullptr, ctx.dispatcher); }
    Image(Image &&o) noexcept : ctx{o.ctx}, image{o.image} { o.image = VK_NULL_HANDLE; }

    vk::Image operator*() const { return image; }
    operator vk::Image() const { return image; }

  protected:
    friend struct Vulkan::VulkanContext;

    Vulkan::VulkanContext &ctx;
    vk::Image image;
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