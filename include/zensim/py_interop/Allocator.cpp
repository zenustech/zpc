#include "zensim/resource/Resource.h"

extern "C" {

zs::memsrc_e mem_enum__host() { return zs::memsrc_e::host; }
zs::memsrc_e mem_enum__device() { return zs::memsrc_e::device; }
zs::memsrc_e mem_enum__um() { return zs::memsrc_e::um; }

zs::ZSPmrAllocator<false> *allocator(zs::memsrc_e mre, zs::ProcID devid) {
  auto ret = new zs::ZSPmrAllocator<false>;
  *ret = get_memory_source(mre, devid);
  return ret;
}
zs::ZSPmrAllocator<true> *allocator_virtual(zs::memsrc_e mre, zs::ProcID devid,
                                            zs::size_t reservedSpace) {
  auto ret = new zs::ZSPmrAllocator<true>;
  *ret = get_virtual_memory_source(mre, devid, reservedSpace, "STACK");
  return ret;
}
}