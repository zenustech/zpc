#pragma once

#if !defined(ZS_ENABLE_VULKAN) || (defined(ZS_ENABLE_VULKAN) && !ZS_ENABLE_VULKAN)
#  error "ZS_ENABLE_VULKAN was not enabled, but Vulkan.hpp was included anyway."
#endif

#include <cstdint>
//
#include "zensim/vulkan/VkBuffer.hpp"
#include "zensim/vulkan/VkContext.hpp"
#include "zensim/vulkan/VkDescriptor.hpp"
#include "zensim/vulkan/VkImage.hpp"
#include "zensim/vulkan/VkPipeline.hpp"
#include "zensim/vulkan/VkRenderPass.hpp"
#include "zensim/vulkan/VkShader.hpp"
#include "zensim/vulkan/VkSwapchain.hpp"
//
#include "zensim/ZpcFunction.hpp"
#include "zensim/container/Callables.hpp"
#include "zensim/types/SourceLocation.hpp"

namespace zs {

  struct Vulkan {
  private:
    Vulkan();

  public:
    ZPC_BACKEND_API static Vulkan &instance() {
      static Vulkan s_instance{};
      return s_instance;
    }
    ~Vulkan();
    void reset();

    static auto &driver() noexcept { return instance(); }
    static size_t num_devices() noexcept { return instance()._contexts.size(); }
    static vk::Instance vk_inst() noexcept { return instance()._instance; }
    static const vk::DispatchLoaderDynamic &vk_inst_dispatcher() noexcept {
      return instance()._dispatcher;
    }
    static auto &context(int devid) { return driver()._contexts[devid]; }
    static auto &context() { return instance()._contexts[instance()._defaultContext]; }

    template <typename F>
    static enable_if_type<is_invocable_r_v<void, F &&>, void> add_instance_destruction_callback(
        F &&f) {
      instance()._onDestroyCallback.insert([cb = FWD(f)]() mutable { cb(); });
    }
    template <typename F>
    static enable_if_type<is_invocable_r_v<void, F &&>, void> set_instance_destruction_callback(
        F &&f) {
      instance()._onDestroyCallback = [cb = FWD(f)]() mutable { cb(); };
    }

  private:
    vk::Instance _instance;
    vk::DispatchLoaderDynamic _dispatcher;  // store vulkan-instance calls
    vk::DebugUtilsMessengerEXT _messenger;
    std::vector<VulkanContext> _contexts;  ///< generally one per device
    zs::callbacks<void()> _onDestroyCallback;
    int _defaultContext = 0;
  };

}  // namespace zs
