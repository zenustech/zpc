#include "VkCommand.hpp"

namespace zs {

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
    if (_waitSemaphores.size() && _stage != nullptr) {
      submitInfo.setPWaitDstStageMask(_stage);
      submitInfo.setWaitSemaphoreCount(_waitSemaphores.size())
          .setPWaitSemaphores(_waitSemaphores.data());
    }
    if (_signalSemaphores.size())
      submitInfo.setSignalSemaphoreCount(_signalSemaphores.size())
          .setPSignalSemaphores(_signalSemaphores.data());

    if (res = _poolFamily.queue.submit(1, &submitInfo, fence, ctx.dispatcher);
        res != vk::Result::eSuccess)
      throw std::runtime_error(fmt::format("failed to submit commands to queue."));

    if (resetConfig) {
      _waitSemaphores.clear();
      _signalSemaphores.clear();
      _stage = nullptr;
    }

    if (_usage == vk_cmd_usage_e::single_use) {
      ctx.device.freeCommandBuffers(_poolFamily.singleUsePool, 1, &_cmd, ctx.dispatcher);
      _cmd = VK_NULL_HANDLE;
    }
  }

}  // namespace zs