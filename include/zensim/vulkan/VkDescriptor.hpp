#pragma once
#include "zensim/vulkan/Vulkan.hpp"

namespace zs {

  struct DescriptorPool {
    DescriptorPool(VulkanContext& ctx) : ctx{ctx}, descriptorPool{VK_NULL_HANDLE} {}
    DescriptorPool(DescriptorPool&& o) noexcept : ctx{o.ctx}, descriptorPool{o.descriptorPool} {
      o.descriptorPool = VK_NULL_HANDLE;
    }
    ~DescriptorPool() { ctx.device.destroyDescriptorPool(descriptorPool, nullptr, ctx.dispatcher); }

    vk::DescriptorPool operator*() const { return descriptorPool; }
    operator vk::DescriptorPool() const { return descriptorPool; }

  protected:
    friend struct VulkanContext;

    VulkanContext& ctx;
    vk::DescriptorPool descriptorPool;
  };

}  // namespace zs