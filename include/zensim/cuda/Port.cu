#include "Cuda.h"
#include "Port.hpp"
#include "zensim/memory/Allocator.h"

namespace zs {

  bool initialize_backend(cuda_exec_tag) {
    (void)Cuda::instance();
    (void)raw_memory_resource<device_mem_tag>::instance();
    (void)raw_memory_resource<um_mem_tag>::instance();
    return true;
  }
  bool deinitialize_backend(cuda_exec_tag) { return true; }

}  // namespace zs