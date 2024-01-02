#pragma once
#include "zensim/vulkan/VkContext.hpp"

namespace zs {

  struct VkCommand {
    using PoolFamily = ExecutionContext::PoolFamily;
    VkCommand(PoolFamily& poolFamily, vk::CommandBuffer cmd, vk_cmd_usage_e usage)
        : _poolFamily{poolFamily},
          _cmd{cmd},
          _usage{usage},
          _stage{nullptr},
          _waitSemaphores{},
          _signalSemaphores{} {}

    VkCommand(VkCommand&& o) noexcept
        : _poolFamily{o._poolFamily},
          _cmd{o._cmd},
          _usage{o._usage},
          _stage{o._stage},
          _waitSemaphores{zs::move(o._waitSemaphores)},
          _signalSemaphores{zs::move(o._signalSemaphores)} {
      o._cmd = VK_NULL_HANDLE;
      o._stage = nullptr;
    }
    ~VkCommand();

    void begin(const vk::CommandBufferBeginInfo& bi) { _cmd.begin(bi); }
    void begin() { _cmd.begin(vk::CommandBufferBeginInfo{usageFlag(), nullptr}); }
    void end() { _cmd.end(); }
    void waitStage(const vk::PipelineStageFlags* stage) { _stage = stage; }
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

    vk::CommandBuffer operator*() const noexcept { return _cmd; }
    operator vk::CommandBuffer() const noexcept { return _cmd; }
    operator vk::CommandBuffer*() noexcept { return &_cmd; }
    operator const vk::CommandBuffer*() const noexcept { return &_cmd; }

  protected:
    friend struct VulkanContext;

    PoolFamily& _poolFamily;
    vk::CommandBuffer _cmd;
    vk_cmd_usage_e _usage;

    const vk::PipelineStageFlags* _stage;
    std::vector<vk::Semaphore> _waitSemaphores, _signalSemaphores;
  };

  struct Fence {
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

    void reset() { _ctx.device.resetFences({_fence}); }

    vk::Fence operator*() const noexcept { return _fence; }
    operator vk::Fence() const noexcept { return _fence; }

  protected:
    friend struct VulkanContext;

    VulkanContext& _ctx;
    vk::Fence _fence;
  };

}  // namespace zs