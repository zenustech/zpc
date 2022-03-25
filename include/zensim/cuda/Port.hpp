#pragma once
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/types/Property.h"

namespace zs {

  ZPC_API bool initialize_backend(cuda_exec_tag);
  ZPC_API bool deinitialize_backend(cuda_exec_tag);

}  // namespace zs