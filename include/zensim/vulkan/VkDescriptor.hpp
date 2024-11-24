#pragma once
#include <map>

#include "zensim/vulkan/VkContext.hpp"

namespace zs {

  struct ZPC_CORE_API DescriptorSetLayout {
    DescriptorSetLayout(VulkanContext& ctx)
        : ctx{ctx}, bindings{}, descriptorSetLayout{VK_NULL_HANDLE} {}
    DescriptorSetLayout(VulkanContext& ctx,
                        const std::map<u32, vk::DescriptorSetLayoutBinding>& descrLayoutBindings)
        : ctx{ctx}, bindings{descrLayoutBindings} {
      std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{};
      for (auto kv : bindings) setLayoutBindings.push_back(kv.second);
      descriptorSetLayout
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
    operator vk::DescriptorSetLayout&() { return descriptorSetLayout; }
    operator const vk::DescriptorSetLayout&() const { return descriptorSetLayout; }

    vk::DescriptorSetLayoutBinding getBinding(u32 binding) const { return bindings.at(binding); }
    auto numBindings() const noexcept { return bindings.size(); }

  protected:
    friend struct VulkanContext;
    friend struct DescriptorWriter;

    VulkanContext& ctx;
    std::map<u32, vk::DescriptorSetLayoutBinding> bindings;
    vk::DescriptorSetLayout descriptorSetLayout;
  };

  class ZPC_CORE_API DescriptorSetLayoutBuilder {
  public:
    DescriptorSetLayoutBuilder(VulkanContext& ctx) noexcept : ctx{ctx} {}
    DescriptorSetLayoutBuilder(DescriptorSetLayoutBuilder&& o) noexcept
        : ctx{o.ctx}, bindings{zs::move(o.bindings)} {}

    DescriptorSetLayoutBuilder& addBinding(u32 binding, vk::DescriptorType descriptorType,
                                           vk::ShaderStageFlags stageFlags, u32 count = 1) {
      bindings[binding]
          = vk::DescriptorSetLayoutBinding{binding, descriptorType, count, stageFlags, nullptr};
      return *this;
    }
    DescriptorSetLayout build() { return DescriptorSetLayout{ctx, bindings}; }

  private:
    VulkanContext& ctx;
    std::map<u32, vk::DescriptorSetLayoutBinding> bindings{};
  };

  struct ZPC_CORE_API DescriptorPool {
    DescriptorPool(VulkanContext& ctx) noexcept : pctx{&ctx}, descriptorPool{VK_NULL_HANDLE} {}
    DescriptorPool(VulkanContext& ctx, const std::vector<vk::DescriptorPoolSize>& poolSizes,
                   u32 maxSets = 1000,
                   vk::DescriptorPoolCreateFlags poolFlags
                   = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
        : pctx{&ctx} {
      descriptorPool = ctx.device.createDescriptorPool(vk::DescriptorPoolCreateInfo{}
                                                           .setPoolSizeCount(poolSizes.size())
                                                           .setPPoolSizes(poolSizes.data())
                                                           .setMaxSets(maxSets)
                                                           .setFlags(poolFlags),
                                                       nullptr, ctx.dispatcher);
    }
    DescriptorPool(const DescriptorPool& o) = delete;
    DescriptorPool(DescriptorPool&& o) noexcept : pctx{o.pctx}, descriptorPool{o.descriptorPool} {
      o.descriptorPool = VK_NULL_HANDLE;
    }
    DescriptorPool& operator=(const DescriptorPool& o) = delete;
    DescriptorPool& operator=(DescriptorPool&& o) noexcept {
      pctx = o.pctx;
      descriptorPool = o.descriptorPool;
      o.descriptorPool = VK_NULL_HANDLE;
      return *this;
    }
    ~DescriptorPool() {
      pctx->device.resetDescriptorPool(descriptorPool, vk::DescriptorPoolResetFlags{},
                                       pctx->dispatcher);
      pctx->device.destroyDescriptorPool(descriptorPool, nullptr, pctx->dispatcher);
    }

    // should not delete this then acquire again for same usage
    void acquireSet(vk::DescriptorSetLayout descriptorSetLayout, vk::DescriptorSet& set) const {
      set = pctx->device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}
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

    VulkanContext* pctx{nullptr};
    vk::DescriptorPool descriptorPool;
  };

  /// @ref little vulkan engine
  struct ZPC_CORE_API DescriptorWriter {
    DescriptorWriter(VulkanContext& ctx, const DescriptorSetLayout& setLayout) noexcept
        : ctx{ctx}, setLayout{setLayout} {}
    // DescriptorWriter(VulkanContext& ctx) noexcept : ctx{ctx}, setLayout{DescriptorSetLayout{ctx}}
    // {}

    DescriptorWriter& writeBuffer(u32 binding, vk::DescriptorBufferInfo* bufferInfo) {
      if (setLayout.bindings.count(binding) != 1)
        throw std::runtime_error("Layout does not contain specified binding");

      const auto& bindingDescription = setLayout.getBinding(binding);
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

      const auto& bindingDescription = setLayout.getBinding(binding);
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

    /// ignore set layout
    DescriptorWriter& writeBuffer(u32 binding, vk::DescriptorType type,
                                  vk::DescriptorBufferInfo* bufferInfo) {
      vk::WriteDescriptorSet write{};
      write.descriptorType = type;
      write.dstBinding = binding;
      write.pBufferInfo = bufferInfo;
      write.descriptorCount = 1;

      writes.push_back(write);
      return *this;
    }
    DescriptorWriter& writeImage(u32 binding, vk::DescriptorType type,
                                 vk::DescriptorImageInfo* imageInfo) {
      vk::WriteDescriptorSet write{};
      write.descriptorType = type;
      write.dstBinding = binding;
      write.pImageInfo = imageInfo;
      write.descriptorCount = 1;

      writes.push_back(write);
      return *this;
    }
    /// @note usually for bindless set
    DescriptorWriter& writeImageI(u32 binding, vk::DescriptorType type, u32 i,
                                  vk::DescriptorImageInfo* imageInfo) {
      vk::WriteDescriptorSet write{};
      write.descriptorType = type;
      write.dstBinding = binding;
      write.dstArrayElement = i;
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
    const DescriptorSetLayout& setLayout;
    std::vector<vk::WriteDescriptorSet> writes;
  };

}  // namespace zs