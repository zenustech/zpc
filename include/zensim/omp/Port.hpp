#pragma once
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/types/Property.h"

namespace zs {

  bool initialize_backend(omp_exec_tag);
  bool deinitialize_backend(omp_exec_tag);

}  // namespace zs