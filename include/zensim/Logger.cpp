#include "Logger.hpp"

namespace zs {

Logger &Logger::instance() noexcept { return s_logger; }
Logger Logger::s_logger{};

}