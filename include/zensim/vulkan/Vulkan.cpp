#include "Vulkan.hpp"

#define VOLK_IMPLEMENTATION
#include "zensim/vulkan/volk/volk.h"
//
#include "zensim/Logger.hpp"
#include "zensim/Platform.hpp"
#include "zensim/types/SourceLocation.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

#define MEM_POOL_CTRL 3

namespace zs {

  /// @ref: https://zhuanlan.zhihu.com/p/634912614
  Vulkan::Vulkan() {
    VkResult r = volkInitialize();
    ZS_ERROR_IF(r == VK_SUCCESS, fmt::format("Unable to complete volk initialization."));
  }

  Vulkan::~Vulkan() {}

}  // namespace zs
