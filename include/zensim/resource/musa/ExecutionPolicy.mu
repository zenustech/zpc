#include "zensim/musa/execution/ExecutionPolicy.muh"

namespace zs {

  ZPC_API ZSPmrAllocator<> get_temporary_memory_source(const MusaExecutionPolicy &pol) {
    ZSPmrAllocator<> ret{};
    ret.res = std::make_unique<temporary_memory_resource<device_mem_tag>>(&pol.context(),
                                                                          pol.getStream());
    ret.location = MemoryLocation{memsrc_e::device, pol.getProcid()};
    ret.cloner = [stream = pol.getStream(), context = &pol.context()]() -> std::unique_ptr<mr_t> {
      std::unique_ptr<mr_t> ret{};
      ret = std::make_unique<temporary_memory_resource<device_mem_tag>>(context, stream);
      return ret;
    };
    return ret;
  }

}  // namespace zs