#pragma once

#include "zensim/vulkan/VkContext.hpp"

namespace zs {
  struct QueryPool {
  public:
    QueryPool() = delete;
    QueryPool(VulkanContext& _ctx) : ctx(_ctx), queryPool(VK_NULL_HANDLE) {}
    QueryPool(QueryPool&& o) noexcept : ctx(o.ctx), queryPool(o.queryPool) {
      o.queryPool = VK_NULL_HANDLE;
    }
    ~QueryPool() {
      ctx.device.destroyQueryPool(queryPool, nullptr, ctx.dispatcher);
      queryPool = VK_NULL_HANDLE;
    }

    vk::QueryPool operator*() const {
      return queryPool;
    }
    operator VkQueryPool() const {
      return queryPool;
    }
    operator vk::QueryPool() const {
      return queryPool;
    }

  protected:
    friend struct VulkanContext;

    VulkanContext& ctx;
    vk::QueryPool queryPool;
  };
}
