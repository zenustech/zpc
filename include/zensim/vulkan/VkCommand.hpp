#pragma once
#include "zensim/vulkan/VkContext.hpp"

namespace zs {

  struct ZPC_CORE_API VkCommand {
    using PoolFamily = ExecutionContext::PoolFamily;
    VkCommand(PoolFamily& poolFamily, vk::CommandBuffer cmd, vk_cmd_usage_e usage);
    VkCommand(VkCommand&& o) noexcept;
    ~VkCommand();

    void begin(const vk::CommandBufferBeginInfo& bi) { _cmd.begin(bi); }
    void begin() { _cmd.begin(vk::CommandBufferBeginInfo{usageFlag(), nullptr}); }
    void end() { _cmd.end(); }
    void waitStage(vk::PipelineStageFlags stageFlag) { _stages = {stageFlag}; }
    void wait(vk::Semaphore s) { _waitSemaphores.push_back(s); }
    void signal(vk::Semaphore s) { _signalSemaphores.push_back(s); }
    void submit(vk::Fence fence, bool resetFence = true, bool resetConfig = false);

    vk::CommandBufferUsageFlags usageFlag() const {
      vk::CommandBufferUsageFlags usageFlags{};
      if (_usage == vk_cmd_usage_e::single_use || _usage == vk_cmd_usage_e::reset)
        usageFlags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
      else
        usageFlags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
      return usageFlags;
    }

    const VulkanContext& ctx() const { return *_poolFamily.pctx; }
    VulkanContext& ctx() { return *_poolFamily.pctx; }
    vk::CommandPool getPool() const noexcept { return _poolFamily.cmdpool(_usage); }
    vk::Queue getQueue() const noexcept { return _poolFamily.queue; }

    vk::CommandBuffer operator*() const noexcept { return _cmd; }
    operator vk::CommandBuffer() const noexcept { return _cmd; }
    operator vk::CommandBuffer*() noexcept { return &_cmd; }
    operator const vk::CommandBuffer*() const noexcept { return &_cmd; }

#if 0
    void swap(VkCommand& r) noexcept {
      zs_swap(_cmd, r._cmd);
      zs_swap(_usage, r._usage);
      zs_swap(_stage, r._stage);
      zs_swap(_waitSemaphores, r._waitSemaphores);
      zs_swap(_signalSemaphores, r._signalSemaphores);
    }
#endif

  protected:
    friend struct VulkanContext;

    PoolFamily& _poolFamily;
    vk::CommandBuffer _cmd;
    vk_cmd_usage_e _usage;

    std::vector<vk::PipelineStageFlags> _stages;
    std::vector<vk::Semaphore> _waitSemaphores, _signalSemaphores;
  };

  struct ZPC_CORE_API Fence {
    Fence(VulkanContext& ctx, bool signaled = false) : _ctx{ctx} {
      vk::FenceCreateInfo ci{};
      if (signaled) ci.setFlags(vk::FenceCreateFlagBits::eSignaled);
      _fence = ctx.device.createFence(ci, nullptr, ctx.dispatcher);
    }

    Fence(Fence&& o) noexcept : _ctx{o._ctx}, _fence{o._fence} { o._fence = VK_NULL_HANDLE; }

    ~Fence() {
      _ctx.device.destroyFence(_fence, nullptr, _ctx.dispatcher);
      _fence = VK_NULL_HANDLE;
    }

    void wait() const;
    void reset() { _ctx.device.resetFences({_fence}); }

    vk::Fence operator*() const noexcept { return _fence; }
    operator vk::Fence() const noexcept { return _fence; }

  protected:
    friend struct VulkanContext;

    VulkanContext& _ctx;
    vk::Fence _fence;
  };

}  // namespace zs