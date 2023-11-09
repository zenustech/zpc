/// reference: taichi/common/core.h
#pragma once

#include "Platform.hpp"
#include "plog/Initializers/RollingFileInitializer.h"
#include "zensim/zpc_tpls/fmt/format.h"
#include "zensim/zpc_tpls/plog/Log.h"

namespace zs {

  // Reference:
  // https://blog.kowalczyk.info/article/j/guide-to-predefined-macros-in-c-compilers-gcc-clang-msvc-etc..html

  struct Logger {
    void log(const int level, const char* fileName, const char* funcName, int line,
             std::string_view msg) {
      PLOG(static_cast<plog::Severity>(level))
          << fmt::format("{}:{}{} {}\n", fileName, funcName, line, msg);
    }
    ZPC_BACKEND_API static Logger& instance();

  private:
    Logger() { plog::init(plog::info, "zensim_logs.log"); }
  };

  ///

#define ZS_LOG(option, ...)                                                                   \
  ::zs::Logger::instance().log(plog::Severity::option, __FILE__, __FUNCTION__, (int)__LINE__, \
                               __VA_ARGS__);

#define ZS_FATAL(...) ZS_LOG(fatal, __VA_ARGS__)
#define ZS_INFO(...) ZS_LOG(info, __VA_ARGS__)
#define ZS_WARN(...) ZS_LOG(warning, __VA_ARGS__)
#define ZS_ERROR(...)           \
  {                             \
    ZS_LOG(error, __VA_ARGS__); \
    ZS_UNREACHABLE;             \
  }

#define ZS_ERROR_IF(condition, ...) \
  if (condition) {                  \
    ZS_ERROR(__VA_ARGS__);          \
  }
#define ZS_WARN_IF(condition, ...) \
  if (condition) {                 \
    ZS_WARN(__VA_ARGS__);          \
  }

}  // namespace zs
