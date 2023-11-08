#include "Logger.hpp"

namespace {
  static zs::Logger *g_loggerInstance = nullptr;
}

namespace zs {

Logger &Logger::instance() {
  if (!g_loggerInstance) g_loggerInstance = new Logger;
  return *g_loggerInstance;
}

}  // namespace zs