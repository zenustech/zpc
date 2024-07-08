#pragma once
#include "zensim/ZpcImplPattern.hpp"
#include "zensim/vulkan/VkContext.hpp"
#include "zensim/vulkan/VkImage.hpp"

namespace zs {

  struct ZPC_CORE_API VkTexture {
    VkTexture() noexcept = default;
    VkTexture(VkTexture &&o) noexcept
        : image{std::move(o.image)},
          sampler{o.sampler},
          imageLayout(o.imageLayout) {
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

    vk::DescriptorImageInfo descriptorInfo() {
      return vk::DescriptorImageInfo{sampler, (vk::ImageView)image.get(), imageLayout};
    }

    Owner<Image> image{};  // including view
    vk::Sampler sampler{VK_NULL_HANDLE};
    vk::ImageLayout imageLayout;
  };

  ZPC_CORE_API VkTexture load_texture(VulkanContext &ctx, u8 *data, size_t numBytes, vk::Extent2D extent,
                         vk::Format format = vk::Format::eR8G8B8A8Unorm,
                         vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal);

}  // namespace zs