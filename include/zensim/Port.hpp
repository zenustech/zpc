#pragma once
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/types/Property.h"

namespace zs {

  bool initialize_backend(host_exec_tag);
  bool deinitialize_backend(host_exec_tag);

}  // namespace zs