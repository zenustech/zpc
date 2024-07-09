#include "zensim/resource/Resource.h"

extern "C" {

ZPC_EXPORT zs::memsrc_e mem_enum__host() { return zs::memsrc_e::host; }
ZPC_EXPORT zs::memsrc_e mem_enum__device() { return zs::memsrc_e::device; }
ZPC_EXPORT zs::memsrc_e mem_enum__um() { return zs::memsrc_e::um; }

ZPC_EXPORT zs::ZSPmrAllocator<false> *allocator(zs::memsrc_e mre, zs::ProcID devid) {
  auto ret = new zs::ZSPmrAllocator<false>;
  *ret = get_memory_source(mre, devid);
  return ret;
}
ZPC_EXPORT zs::ZSPmrAllocator<true> *allocator_virtual(zs::memsrc_e mre, zs::ProcID devid,
                                                       zs::size_t reservedSpace) {
  auto ret = new zs::ZSPmrAllocator<true>;
  *ret = get_virtual_memory_source(mre, devid, reservedSpace, "STACK");
  return ret;
}
ZPC_EXPORT void del_allocator(zs::ZSPmrAllocator<false> *allocator) { delete allocator; }
ZPC_EXPORT void del_allocator_virtual(zs::ZSPmrAllocator<false> *allocator) { delete allocator; }
}