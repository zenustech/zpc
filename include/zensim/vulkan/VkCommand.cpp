#include "VkCommand.hpp"

#include "zensim/ZpcFunctional.hpp"

namespace zs {

  void Fence::wait() const {
    if (_ctx.device.waitForFences({(vk::Fence)_fence}, VK_TRUE, detail::deduce_numeric_max<u64>(),
                                  _ctx.dispatcher)
        != vk::Result::eSuccess)
      throw std::runtime_error("error waiting for fences");
  }

  VkCommand::VkCommand(PoolFamily& poolFamily, vk::CommandBuffer cmd, vk_cmd_usage_e usage)
      : _poolFamily{poolFamily},
        _cmd{cmd},
        _usage{usage},
        _stages{},
        _waitSemaphores{},
        _signalSemaphores{} {}

  VkCommand::VkCommand(VkCommand&& o) noexcept
      : _poolFamily{o._poolFamily},
        _cmd{zs::exchange(o._cmd, VK_NULL_HANDLE)},
        _usage{o._usage},
        _stages{zs::move(o._stages)},
        _waitSemaphores{zs::move(o._waitSemaphores)},
        _signalSemaphores{zs::move(o._signalSemaphores)} {
    o._cmd = VK_NULL_HANDLE;
  }
  VkCommand::~VkCommand() {
    if (_cmd != VK_NULL_HANDLE) {
      auto& c = this->ctx();
      c.device.freeCommandBuffers(getPool(), 1, &_cmd, c.dispatcher);
      _cmd = VK_NULL_HANDLE;
    }
  }
  void VkCommand::submit(vk::Fence fence, bool resetFence, bool resetConfig) {
    if (_cmd == VK_NULL_HANDLE) return;

    const auto& ctx = this->ctx();

    vk::Result res;
    if (resetFence)
      if (res = ctx.device.resetFences(1, &fence); res != vk::Result::eSuccess)
        throw std::runtime_error(fmt::format("failed to reset buffered fences."));

    auto submitInfo = vk::SubmitInfo().setCommandBufferCount(1).setPCommandBuffers(&_cmd);
    if (!_stages.empty())
      submitInfo.setWaitDstStageMask(_stages);
    if (_waitSemaphores.size())
      submitInfo.setWaitSemaphoreCount(_waitSemaphores.size())
          .setPWaitSemaphores(_waitSemaphores.data());
    if (_signalSemaphores.size())
      submitInfo.setSignalSemaphoreCount(_signalSemaphores.size())
          .setPSignalSemaphores(_signalSemaphores.data());

    if (res = _poolFamily.queue.submit(1, &submitInfo, fence, ctx.dispatcher);
        res != vk::Result::eSuccess)
      throw std::runtime_error(fmt::format("failed to submit commands to queue."));

    if (resetConfig) {
      _waitSemaphores.clear();
      _signalSemaphores.clear();
      _stages.clear();
    }

    if (_usage == vk_cmd_usage_e::single_use) {
      if (ctx.device.waitForFences({(vk::Fence)fence}, VK_TRUE, detail::deduce_numeric_max<u64>(),
                                   ctx.dispatcher)
          != vk::Result::eSuccess)
        throw std::runtime_error("error waiting for fences");
      ctx.device.freeCommandBuffers(_poolFamily.singleUsePool, 1, &_cmd, ctx.dispatcher);
      _cmd = VK_NULL_HANDLE;
    }
  }

}  // namespace zs