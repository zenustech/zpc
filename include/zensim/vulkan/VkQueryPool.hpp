#pragma once

#include "zensim/vulkan/VkContext.hpp"

namespace zs {
  struct QueryPool {
  public:
    QueryPool(VulkanContext& ctx)
        : ctx(ctx), queryPool{VK_NULL_HANDLE}, queryType{}, queryCount{0} {}
    QueryPool(QueryPool&& o) noexcept
        : ctx(o.ctx),
          queryPool{zs::exchange(o.queryPool, VK_NULL_HANDLE)},
          queryType{zs::exchange(o.queryType, {})},
          queryCount{zs::exchange(o.queryCount, 0)} {}
    ~QueryPool() {
      if (queryPool) {
        ctx.device.destroyQueryPool(queryPool, nullptr, ctx.dispatcher);
        queryPool = VK_NULL_HANDLE;
        queryCount = 0;
      }
    }

    u32 getCount() const noexcept { return queryCount; }

    vk::QueryPool operator*() const { return queryPool; }
    operator VkQueryPool() const { return queryPool; }
    operator vk::QueryPool() const { return queryPool; }

  protected:
    friend struct VulkanContext;

    VulkanContext& ctx;
    vk::QueryPool queryPool;
    vk::QueryType queryType;
    u32 queryCount;
  };
}  // namespace zs
