#pragma once
#include "zensim/vulkan/Vulkan.hpp"

namespace zs {

  struct DescriptorSetLayout {
    DescriptorSetLayout(VulkanContext& ctx) : ctx{ctx}, descriptorSetLayout{VK_NULL_HANDLE} {}
    DescriptorSetLayout(VulkanContext& ctx,
                        const std::vector<vk::DescriptorSetLayoutBinding>& descrLayoutBindings)
        : ctx{ctx} {
      descriptorSetLayout
          = ctx.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{}
                                                     .setBindingCount(descrLayoutBindings.size())
                                                     .setPBindings(descrLayoutBindings.data()),
                                                 nullptr, ctx.dispatcher);
    }
    DescriptorSetLayout(DescriptorSetLayout&& o) noexcept
        : ctx{o.ctx}, descriptorSetLayout{o.descriptorSetLayout} {
      o.descriptorSetLayout = VK_NULL_HANDLE;
    }
    ~DescriptorSetLayout() {
      ctx.device.destroyDescriptorSetLayout(descriptorSetLayout, nullptr, ctx.dispatcher);
    }

    vk::DescriptorSetLayout operator*() const { return descriptorSetLayout; }
    operator vk::DescriptorSetLayout() const { return descriptorSetLayout; }

  protected:
    friend struct VulkanContext;

    VulkanContext& ctx;
    vk::DescriptorSetLayout descriptorSetLayout;
  };

  struct DescriptorPool {
    DescriptorPool(VulkanContext& ctx) noexcept : ctx{ctx}, descriptorPool{VK_NULL_HANDLE} {}
    DescriptorPool(VulkanContext& ctx, const std::vector<vk::DescriptorPoolSize>& poolSizes,
                   u32 maxSets = 1000,
                   vk::DescriptorPoolCreateFlags poolFlags
                   = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
        : ctx{ctx} {
      descriptorPool = ctx.device.createDescriptorPool(vk::DescriptorPoolCreateInfo{}
                                                           .setPoolSizeCount(poolSizes.size())
                                                           .setPPoolSizes(poolSizes.data())
                                                           .setMaxSets(maxSets)
                                                           .setFlags(poolFlags),
                                                       nullptr, ctx.dispatcher);
    }
    DescriptorPool(DescriptorPool&& o) noexcept : ctx{o.ctx}, descriptorPool{o.descriptorPool} {
      o.descriptorPool = VK_NULL_HANDLE;
    }
    ~DescriptorPool() { ctx.device.destroyDescriptorPool(descriptorPool, nullptr, ctx.dispatcher); }

    // should not delete this then acquire again for same usage
    void acquireSet(vk::DescriptorSetLayout descriptorSetLayout, vk::DescriptorSet& set) const {
      set = ctx.device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}
                                                  .setDescriptorPool(descriptorPool)
                                                  .setPSetLayouts(&descriptorSetLayout)
                                                  .setDescriptorSetCount(1))[0];
      /// @note from lve
      // Might want to create a "DescriptorPoolManager" class that handles this case, and builds
      // a new pool whenever an old pool fills up. But this is beyond our current scope
    }

    vk::DescriptorPool operator*() const { return descriptorPool; }
    operator vk::DescriptorPool() const { return descriptorPool; }

  protected:
    friend struct VulkanContext;

    VulkanContext& ctx;
    vk::DescriptorPool descriptorPool;
  };

}  // namespace zs