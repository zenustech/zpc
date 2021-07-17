/// reference: taichi/common/core.h
#pragma once

#include "zensim/Singleton.h"
#include "zensim/tpls/fmt/core.h"
#include "zensim/tpls/loguru/loguru.hpp"

namespace zs {

  // Reference:
  // https://blog.kowalczyk.info/article/j/guide-to-predefined-macros-in-c-compilers-gcc-clang-msvc-etc..html

  struct Logger : Singleton<Logger> {
    Logger() {
      loguru::add_file("zensim_logs.log", loguru::Append, loguru::NamedVerbosity::Verbosity_MAX);
    }
    void log(const int level, const char* fileName, const char* funcName, int line,
             std::string_view msg) {
      VLOG_F(level, "[%s:%s@%d] %s", fileName, funcName, line, msg.data());
    }
  };

///

#  define ZS_LOG(option, ...)                                                                 \
    Logger::instance().log(loguru::Verbosity_##option, __FILE__, __FUNCTION__, (int)__LINE__, \
                           __VA_ARGS__);
  // LOG_F(option, "[%s:%s@%d] %s", __FILE__, __FUNCTION__, (int)__LINE__, __VA_ARGS__)

#define ZS_FATAL(...) ZS_LOG(FATAL, __VA_ARGS__)
#define ZS_INFO(...) ZS_LOG(INFO, __VA_ARGS__)
#define ZS_WARN(...) ZS_LOG(WARNING, __VA_ARGS__)
#define ZS_ERROR(...)           \
  {                             \
    ZS_LOG(ERROR, __VA_ARGS__); \
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
