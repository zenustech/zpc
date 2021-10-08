#include "Resource.h"

#include "zensim/execution/Concurrency.h"
#include "zensim/memory/Allocator.h"
#include "zensim/memory/MemoryResource.h"
#if defined(ZS_ENABLE_CUDA)
#  include "zensim/cuda/memory/Allocator.h"
#endif

namespace zs {

  static std::shared_mutex g_resource_rw_mutex{};
  static concurrent_map<void *, Resource::AllocationRecord> g_resource_records;

  void record_allocation(mem_tags tag, void *ptr, std::string_view name, std::size_t size,
                         std::size_t alignment) {
    get_resource_manager().record(tag, ptr, name, size, alignment);
  }
  void erase_allocation(void *ptr) { get_resource_manager().erase(ptr); }

  void copy(MemoryEntity dst, MemoryEntity src, std::size_t size) {
    if (dst.location.onHost() && src.location.onHost())
      copy(mem_host, dst.ptr, src.ptr, size);
    else
      copy(mem_device, dst.ptr, src.ptr, size);
  }

  ZSPmrAllocator<> get_memory_source(memsrc_e mre, ProcID devid, std::string_view advice) {
    const mem_tags tag = to_memory_source_tag(mre);
    ZSPmrAllocator<> ret{};
    if (advice.empty()) {
      if (mre == memsrc_e::um) {
        if (devid < -1)
          match([&ret, devid](auto &tag) {
            ret.setOwningUpstream<advisor_memory_resource>(tag, devid, "READ_MOSTLY");
          })(tag);
        else
          match([&ret, devid](auto &tag) {
            ret.setOwningUpstream<advisor_memory_resource>(tag, devid, "PREFERRED_LOCATION");
          })(tag);
      } else
        // match([&ret](auto &tag) { ret.setNonOwningUpstream<raw_memory_resource>(tag); })(tag);
        match([&ret, devid](auto &tag) {
          ret.setOwningUpstream<default_memory_resource>(tag, devid);
        })(tag);
      // ret.setNonOwningUpstream<raw_memory_resource>(tag);
    } else
      match([&ret, &advice, devid](auto &tag) {
        ret.setOwningUpstream<advisor_memory_resource>(tag, devid, advice);
      })(tag);
    return ret;
  }
  ZSPmrAllocator<true> get_virtual_memory_source(memsrc_e mre, ProcID devid, std::size_t bytes,
                                                 std::string_view option) {
    const mem_tags tag = to_memory_source_tag(mre);
    ZSPmrAllocator<true> ret{};
    if (mre == memsrc_e::um)
      throw std::runtime_error("no corresponding virtual memory resource for [um]");
    match(
        [&ret, devid, bytes,
         option](auto tag) -> std::enable_if_t<!is_same_v<decltype(tag), um_mem_tag>> {
          if (option == "ARENA")
            ret.setOwningUpstream<arena_virtual_memory_resource>(tag, devid, bytes);
#if 0
          else if (option == "STACK" || option.empty())
            ret.setOwningUpstream<stack_virtual_memory_resource>(tag, devid, bytes);
          else
            throw std::runtime_error(fmt::format("unkonwn vmr option [{}]\n", option));
#endif
        },
        [](...) {})(tag);
    return ret;
  }

  Resource::~Resource() {
    for (auto &&record : g_resource_records) {
      const auto &[ptr, info] = record;
      fmt::print("recycling allocation [{}], tag [{}], size [{}], alignment [{}], allocator [{}]\n",
                 (std::uintptr_t)ptr,
                 match([](auto &tag) { return get_memory_tag_name(tag); })(info.tag), info.size,
                 info.alignment, info.allocatorType);
    }
  }
  void Resource::record(mem_tags tag, void *ptr, std::string_view name, std::size_t size,
                        std::size_t alignment) {
    g_resource_records.set(ptr, AllocationRecord{tag, size, alignment, std::string(name)});
  }
  void Resource::erase(void *ptr) { g_resource_records.erase(ptr); }

  void Resource::deallocate(void *ptr) {
    if (g_resource_records.find(ptr) != nullptr) {
      std::unique_lock lk(g_resource_rw_mutex);
      const auto &r = g_resource_records.get(ptr);
      match([&r, ptr](auto &tag) { zs::deallocate(tag, ptr, r.size, r.alignment); })(r.tag);
    } else
      throw std::runtime_error(
          fmt::format("allocation record {} not found in records!", (std::uintptr_t)ptr));
    g_resource_records.erase(ptr);
  }

  Resource &get_resource_manager() noexcept { return Resource::instance(); }

}  // namespace zs
