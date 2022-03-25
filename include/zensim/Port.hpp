#pragma once
#include "Platform.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/types/Property.h"

namespace zs {

  ZPC_API bool initialize_backend(host_exec_tag);
  ZPC_API bool deinitialize_backend(host_exec_tag);

}  // namespace zs