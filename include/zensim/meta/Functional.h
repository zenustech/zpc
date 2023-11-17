#pragma once

#include <limits>

#include "../ZpcFunctional.hpp"
#include "Meta.h"

namespace zs {

  // gcem alike, shorter alias for std::numeric_limits
  template <typename T> using limits = std::numeric_limits<T>;

}  // namespace zs
