#pragma once
#include "zensim/vulkan/Vulkan.hpp"

namespace zs {

  struct DescriptorSetLayout {
    DescriptorSetLayout(VulkanContext& ctx)
        : ctx{ctx}, bindings{}, descriptorSetLayout{VK_NULL_HANDLE} {}
    DescriptorSetLayout(VulkanContext& ctx,
                        const std::map<u32, vk::DescriptorSetLayoutBinding>& descrLayoutBindings)
        : ctx{ctx}, bindings{descrLayoutBindings} {
      std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{};
      for (auto kv : bindings) setLayoutBindings.push_back(kv.second);
      auto descriptorSetLayout
          = ctx.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{}
                                                     .setBindingCount(setLayoutBindings.size())
                                                     .setPBindings(setLayoutBindings.data()),
                                                 nullptr, ctx.dispatcher);
    }
    DescriptorSetLayout(DescriptorSetLayout&& o) noexcept
        : ctx{o.ctx}, bindings{std::move(o.bindings)}, descriptorSetLayout{o.descriptorSetLayout} {
      o.descriptorSetLayout = VK_NULL_HANDLE;
    }
    ~DescriptorSetLayout() {
      ctx.device.destroyDescriptorSetLayout(descriptorSetLayout, nullptr, ctx.dispatcher);
    }

    vk::DescriptorSetLayout operator*() const { return descriptorSetLayout; }
    operator vk::DescriptorSetLayout() const { return descriptorSetLayout; }

  protected:
    friend struct VulkanContext;
    friend struct DescriptorWriter;

    VulkanContext& ctx;
    std::map<u32, vk::DescriptorSetLayoutBinding> bindings;
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
    ~DescriptorPool() {
      ctx.device.resetDescriptorPool(descriptorPool, vk::DescriptorPoolResetFlags{},
                                     ctx.dispatcher);
      ctx.device.destroyDescriptorPool(descriptorPool, nullptr, ctx.dispatcher);
    }

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

  /// @ref little vulkan engine
  struct DescriptorWriter {
    DescriptorWriter(VulkanContext& ctx, DescriptorSetLayout& setLayout,
                     DescriptorPool& pool) noexcept
        : ctx{ctx}, setLayout{setLayout}, pool{pool} {}

    DescriptorWriter& writeBuffer(u32 binding, vk::DescriptorBufferInfo* bufferInfo) {
      if (setLayout.bindings.count(binding) != 1)
        throw std::runtime_error("Layout does not contain specified binding");

      const auto& bindingDescription = setLayout.bindings[binding];
      if (bindingDescription.descriptorCount != 1)
        throw std::runtime_error("Binding single descriptor info, but binding expects multiple.");

      vk::WriteDescriptorSet write{};
      write.descriptorType = bindingDescription.descriptorType;
      write.dstBinding = binding;
      write.pBufferInfo = bufferInfo;
      write.descriptorCount = 1;

      writes.push_back(write);
      return *this;
    }
    DescriptorWriter& writeImage(u32 binding, vk::DescriptorImageInfo* imageInfo) {
      if (setLayout.bindings.count(binding) != 1)
        throw std::runtime_error("Layout does not contain specified binding");

      const auto& bindingDescription = setLayout.bindings[binding];
      if (bindingDescription.descriptorCount != 1)
        throw std::runtime_error("Binding single descriptor info, but binding expects multiple.");

      vk::WriteDescriptorSet write{};
      write.descriptorType = bindingDescription.descriptorType;
      write.dstBinding = binding;
      write.pImageInfo = imageInfo;
      write.descriptorCount = 1;

      writes.push_back(write);
      return *this;
    }

    void overwrite(vk::DescriptorSet& set) {
      for (auto& write : writes) write.dstSet = set;
      ctx.device.updateDescriptorSets(/*WriteDescriptorSet*/ writes.size(), writes.data(),
                                      /*CopyDescriptorSet*/ 0, nullptr, ctx.dispatcher);
    }

  private:
    friend struct VulkanContext;

    VulkanContext& ctx;
    DescriptorSetLayout& setLayout;
    DescriptorPool& pool;
    std::vector<vk::WriteDescriptorSet> writes;
  };

}  // namespace zs