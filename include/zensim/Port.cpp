#include "Port.hpp"

#include "zensim/memory/Allocator.h"

namespace zs {

  bool initialize_backend(host_exec_tag) { return true; }
  bool deinitialize_backend(host_exec_tag) { return true; }

}  // namespace zs