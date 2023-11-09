#include "zensim/Zpc.hpp"
#include "zensim/execution/ConcurrencyPrimitive.hpp"

#include "zensim/io/IO.h"
#if ZS_ENABLE_CUDA
#  include "zensim/cuda/Cuda.h"
#endif
#if ZS_ENABLE_VULKAN
#  include "zensim/vulkan/Vulkan.hpp"
#endif
#include "zensim/Logger.hpp"

namespace {
  static zs::Mutex g_zpcInitMutex;
  static std::atomic<bool> g_isZpcInitialized = false;
}  // namespace

namespace zs {

  void initialize() {
    if (g_isZpcInitialized.load(std::memory_order_acquire)) return;
    g_zpcInitMutex.lock();
    if (g_isZpcInitialized.load(std::memory_order_acquire)) return;

    (void)IO::instance();

#if ZS_ENABLE_CUDA
    (void)Cuda::instance();
#endif

#if ZS_ENABLE_VULKAN
    (void)Vulkan::instance();
#endif

    g_isZpcInitialized.store(true, std::memory_order_release);
    g_zpcInitMutex.unlock();
  }

}  // namespace zs