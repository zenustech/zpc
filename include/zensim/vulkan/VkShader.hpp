#pragma once
#include "zensim/vulkan/VkContext.hpp"

namespace zs {

  struct ShaderModule {
    ShaderModule(VulkanContext& ctx) : ctx{ctx}, shaderModule{VK_NULL_HANDLE} {}
    ShaderModule(ShaderModule&& o) noexcept : ctx{o.ctx}, shaderModule{o.shaderModule} {
      o.shaderModule = VK_NULL_HANDLE;
    }
    ~ShaderModule() { ctx.device.destroyShaderModule(shaderModule, nullptr, ctx.dispatcher); }

    vk::ShaderModule operator*() const { return shaderModule; }
    operator vk::ShaderModule() const { return shaderModule; }

  protected:
    friend struct VulkanContext;

    VulkanContext& ctx;
    vk::ShaderModule shaderModule;
  };

}  // namespace zs