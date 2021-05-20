#include "Resource.h"

namespace zs {

  void record_allocation(mem_tags tag, void *ptr, std::string_view name, std::size_t size,
                         std::size_t alignment) {
    get_resource_manager().record(tag, ptr, name, size, alignment);
  }
  void erase_allocation(void *ptr) { get_resource_manager().erase(ptr); }

  void copy(MemoryEntity dst, MemoryEntity src, std::size_t size) {
    if (dst.descr.onHost() && src.descr.onHost())
      copy(mem_host, dst.ptr, src.ptr, size);
    else
      copy(mem_device, dst.ptr, src.ptr, size);
  }

  GeneralAllocator get_memory_source(memsrc_e mre, ProcID devid, std::string_view advice) {
    const mem_tags tag = to_memory_source_tag(mre);
    GeneralAllocator ret{};
    if (advice.empty()) {
      if (mre == memsrc_e::um) {
        if (devid < -1)
          match([&ret, devid](auto &tag) {
            ret.construct<advisor_allocator>(tag, "READ_MOSTLY", devid);
          })(tag);
        else
          match([&ret, devid](auto &tag) {
            ret.construct<advisor_allocator>(tag, "PREFERRED_LOCATION", devid);
          })(tag);
      } else
        match([&ret](auto &tag) { ret.construct<raw_allocator>(tag); })(tag);
      // ret.construct<raw_allocator>(tag);
    } else
      match([&ret, &advice, devid](auto &tag) {
        ret.construct<advisor_allocator>(tag, advice, devid);
      })(tag);
    return ret;
  }

  void Resource::record(mem_tags tag, void *ptr, std::string_view name, std::size_t size,
                        std::size_t alignment) {
    _records.set(ptr, AllocationRecord{tag, size, alignment, std::string(name)});
  }
  void Resource::erase(void *ptr) { _records.erase(ptr); }

  void Resource::deallocate(void *ptr) {
    if (_records.find(ptr) != nullptr) {
      std::unique_lock lk(_rw);
      const auto &r = _records.get(ptr);
      match([&r, ptr](auto &tag) { zs::deallocate(tag, ptr, r.size, r.alignment); })(r.tag);
    } else
      throw std::runtime_error(
          fmt::format("allocation record {} not found in records!", (std::uintptr_t)ptr));
    _records.erase(ptr);
  }

  Resource &get_resource_manager() noexcept { return Resource::instance(); }

}  // namespace zs
