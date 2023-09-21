#pragma once
#include "zensim/vulkan/VkContext.hpp"
#include "zensim/vulkan/VkDescriptor.hpp"

namespace zs {

  struct ShaderModule {
    ShaderModule(VulkanContext& ctx)
        : ctx{ctx},
          setLayouts{},
          compiled{nullptr, nullptr},
          resources{nullptr, nullptr},
          shaderModule{VK_NULL_HANDLE},
          stageFlag{} {}
    ShaderModule(ShaderModule&& o) noexcept
        : ctx{o.ctx},
          setLayouts{std::move(o.setLayouts)},
          compiled{std::move(o.compiled)},
          resources{std::move(o.resources)},
          shaderModule{o.shaderModule},
          stageFlag{o.stageFlag} {
      o.shaderModule = VK_NULL_HANDLE;
      o.stageFlag = {};
    }
    ~ShaderModule() { ctx.device.destroyShaderModule(shaderModule, nullptr, ctx.dispatcher); }

    void displayLayoutInfo();
    const std::map<u32, DescriptorSetLayout>& layouts() const noexcept { return setLayouts; }
    const DescriptorSetLayout& layout(u32 no = 0) const { return setLayouts.at(no); }
    DescriptorSetLayout& layout(u32 no = 0) { return setLayouts.at(no); }

    vk::ShaderModule operator*() const { return shaderModule; }
    operator vk::ShaderModule() const { return shaderModule; }

  protected:
    friend struct VulkanContext;
    void analyzeLayout(const u32* code, size_t size);
    void initializeDescriptorSetLayouts();

    VulkanContext& ctx;
    //
    std::map<u32, DescriptorSetLayout> setLayouts;  // descriptor set layouts
    std::unique_ptr<void, void (*)(void const*)> compiled, resources;
    // inherent data
    vk::ShaderModule shaderModule;
    vk::ShaderStageFlagBits stageFlag;
  };

}  // namespace zs