#pragma once
#include "zensim/ZpcImplPattern.hpp"
#include "zensim/vulkan/VkContext.hpp"
#include "zensim/vulkan/VkImage.hpp"

namespace zs {

  struct VkTexture {
    VkTexture() noexcept = default;
    VkTexture(VkTexture &&o) noexcept
        : image{std::move(o.image)},
          sampler{o.sampler},
          imageLayout(o.imageLayout),
          width{o.width},
          height{o.height},
          mipLevels(o.mipLevels) {
      o.sampler = VK_NULL_HANDLE;
    }
    VkTexture &operator=(VkTexture &&o) {
      if (image) {
        auto &ctx = image.get().ctx;
        ctx.device.destroySampler(sampler, nullptr, ctx.dispatcher);
      }
      image = zs::move(o.image);
      sampler = o.sampler;
      o.sampler = VK_NULL_HANDLE;
      imageLayout = o.imageLayout;
      width = o.width;
      height = o.height;
      mipLevels = o.mipLevels;
      return *this;
    }
    VkTexture &operator=(const VkTexture &o) = delete;
    VkTexture(const VkTexture &o) = delete;

    ~VkTexture() { reset(); }
    void reset() {
      if (image) {
        auto &ctx = image.get().ctx;
        ctx.device.destroySampler(sampler, nullptr, ctx.dispatcher);
        image.reset();
      }
    }
    explicit operator bool() const noexcept { return static_cast<bool>(image); }

    Owner<Image> image{};  // including view
    vk::Sampler sampler{VK_NULL_HANDLE};
    vk::ImageLayout imageLayout;
    u32 width, height;
    u32 mipLevels;
  };

  VkTexture load_texture(VulkanContext &ctx, u8 *data, size_t numBytes, vk::Extent2D extent,
                         vk::Format format = vk::Format::eR8G8B8A8Unorm,
                         vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal);

}  // namespace zs