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

}  // namespace zs