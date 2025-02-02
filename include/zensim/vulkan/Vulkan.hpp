#pragma once

#if !defined(ZS_ENABLE_VULKAN) || (defined(ZS_ENABLE_VULKAN) && !ZS_ENABLE_VULKAN)
#  error "ZS_ENABLE_VULKAN was not enabled, but Vulkan.hpp was included anyway."
#endif

//
#include "zensim/vulkan/VkBuffer.hpp"
#include "zensim/vulkan/VkCommand.hpp"
#include "zensim/vulkan/VkContext.hpp"
#include "zensim/vulkan/VkDescriptor.hpp"
#include "zensim/vulkan/VkImage.hpp"
#include "zensim/vulkan/VkQueryPool.hpp"
// #include "zensim/vulkan/VkModel.hpp"
#include "zensim/vulkan/VkPipeline.hpp"
#include "zensim/vulkan/VkRenderPass.hpp"
#include "zensim/vulkan/VkShader.hpp"
#include "zensim/vulkan/VkSwapchain.hpp"
#include "zensim/vulkan/VkTexture.hpp"
//
#include "zensim/ZpcFunction.hpp"
#include "zensim/container/Callables.hpp"
#include "zensim/types/SourceLocation.hpp"

namespace zs {

  struct ZPC_CORE_API Vulkan {
  private:
    Vulkan();

    Vulkan(Vulkan &&) = delete;
    Vulkan &operator=(Vulkan &&) = delete;
    Vulkan(const Vulkan &) = delete;
    Vulkan &operator=(const Vulkan &) = delete;

  public:
    static Vulkan &instance();
    ~Vulkan();
    void reset();

    static Vulkan &driver() noexcept;
    static size_t num_devices() noexcept;
    static vk::Instance vk_inst() noexcept;
    static const ZS_VK_DISPATCH_LOADER_DYNAMIC &vk_inst_dispatcher() noexcept;
    static VulkanContext &context(int devid);
    static VulkanContext &context();

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

    friend struct VulkanContext;

    template <typename T> T &working_contexts() { return *static_cast<T *>(_workingContexts); }
    template <typename T> T &mutex() { return *static_cast<T *>(_mutex); }

  private:
    vk::Instance _instance;
    ZS_VK_DISPATCH_LOADER_DYNAMIC _dispatcher;  // store vulkan-instance calls
    vk::DebugUtilsMessengerEXT _messenger;
    std::vector<VulkanContext> _contexts;  ///< generally one per device
    zs::callbacks<void()> _onDestroyCallback;
    int _defaultContext = 0;

    void *_workingContexts, *_mutex;
  };

}  // namespace zs
