#pragma once
#include "zensim/vulkan/VkContext.hpp"

namespace zs {

  struct ShaderModule {
    ShaderModule(VulkanContext& ctx)
        : ctx{ctx},
          compiled{nullptr, nullptr},
          resources{nullptr, nullptr},
          shaderModule{VK_NULL_HANDLE} {}
    ShaderModule(ShaderModule&& o) noexcept
        : ctx{o.ctx},
          compiled{nullptr, nullptr},
          resources{nullptr, nullptr},
          shaderModule{o.shaderModule} {
      o.shaderModule = VK_NULL_HANDLE;
    }
    ~ShaderModule() { ctx.device.destroyShaderModule(shaderModule, nullptr, ctx.dispatcher); }

    vk::ShaderModule operator*() const { return shaderModule; }
    operator vk::ShaderModule() const { return shaderModule; }

  protected:
    friend struct VulkanContext;
    void analyzeLayout(const u32* code, size_t size);

    VulkanContext& ctx;
    std::unique_ptr<void, void (*)(void const*)> compiled, resources;
    vk::ShaderModule shaderModule;
  };

}  // namespace zs