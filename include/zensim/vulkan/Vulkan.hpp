#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
//
#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.hpp"
//
#include "zensim/Reflection.h"
#include "zensim/Singleton.h"
#include "zensim/profile/CppTimers.hpp"
#include "zensim/types/SourceLocation.hpp"
#include "zensim/types/Tuple.h"
#include "zensim/zpc_tpls/fmt/format.h"

namespace zs {

  struct Vulkan : Singleton<Vulkan> {
  public:
    Vulkan();
    ~Vulkan();

    static auto &driver() noexcept { return instance(); }
    static auto &context(int devid) { return driver().contexts[devid]; }

    struct VulkanContext {
      auto &driver() const noexcept { return Vulkan::driver(); }
      VulkanContext(int devId = 0) : devid{devId} {}
      auto getDevId() const noexcept { return devid; }

      int devid;
      vk::Device device;
    };

  private:
    std::vector<VulkanContext> contexts;  ///< generally one per device
  };

}  // namespace zs
