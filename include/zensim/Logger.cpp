#include "Logger.hpp"
#include "zensim/execution/ConcurrencyPrimitive.hpp"

namespace {
  static zs::Mutex g_loggerMutex;
  static std::atomic<bool> g_isLoggerInitialized = false;
  static zs::Logger *g_loggerInstance = nullptr;
}

namespace zs {

Logger &Logger::instance() {
  if (g_isLoggerInitialized.load(std::memory_order_acquire)) return *g_loggerInstance;
  g_loggerMutex.lock();
  if (g_isLoggerInitialized.load(std::memory_order_acquire)) return *g_loggerInstance;

  if (!g_loggerInstance) g_loggerInstance = new Logger;

  g_isLoggerInitialized.store(true, std::memory_order_release);
  g_loggerMutex.unlock();
  return *g_loggerInstance;
}

}  // namespace zs