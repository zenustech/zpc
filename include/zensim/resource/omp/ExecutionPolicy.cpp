#include "zensim/omp/execution/ExecutionPolicy.hpp"

namespace zs {

  ZPC_API ZSPmrAllocator<> get_temporary_memory_source(const OmpExecutionPolicy &pol) {
    return get_memory_source(memsrc_e::host, (ProcID)-1);
  }

}  // namespace zs