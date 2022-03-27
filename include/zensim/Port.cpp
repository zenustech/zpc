#include "Port.hpp"

#include "zensim/memory/Allocator.h"
#include "zensim/Logger.hpp"

namespace zs {

  bool initialize_backend(host_exec_tag) {
    (void)Logger::instance();
    (void)raw_memory_resource<host_mem_tag>::instance();
    return true;
  }
  bool deinitialize_backend(host_exec_tag) { return true; }

}  // namespace zs