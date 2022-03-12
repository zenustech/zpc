#include "Port.hpp"

#include "execution/ExecutionPolicy.hpp"
#include "zensim/memory/Allocator.h"

namespace zs {

  bool initialize_backend(omp_exec_tag) { return true; }
  bool deinitialize_backend(omp_exec_tag) { return true; }

}  // namespace zs