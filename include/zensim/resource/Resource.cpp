#include "Resource.h"

#include "zensim/Port.hpp"
#include "zensim/execution/Concurrency.h"
#include "zensim/memory/MemoryResource.h"
#if ZS_ENABLE_CUDA
#  include "zensim/cuda/Port.hpp"
#endif
#if ZS_ENABLE_OPENMP
#  include "zensim/omp/Port.hpp"
#endif

namespace zs {

  static std::shared_mutex g_resource_rw_mutex{};
  static concurrent_map<void *, Resource::AllocationRecord> g_resource_records;

#if 1
  static Resource g_resource;
  Resource &Resource::instance() noexcept { return g_resource; }
#else
  Resource &Resource::instance() noexcept { 
    static Resource *ptr = new Resource();
    return *ptr; 
  }
#endif
  std::atomic_ullong &Resource::counter() noexcept { return instance()._counter; }

  Resource::Resource() {
    initialize_backend(exec_seq);
#if ZS_ENABLE_CUDA
    puts("cuda initialized");
    initialize_backend(exec_cuda);
#endif
#if ZS_ENABLE_OPENMP
    puts("openmp initialized");
    initialize_backend(exec_omp);
#endif
    // sycl...
  }
  Resource::~Resource() {
    for (auto &&record : g_resource_records) {
      const auto &[ptr, info] = record;
      fmt::print("recycling allocation [{}], tag [{}], size [{}], alignment [{}], allocator [{}]\n",
                 (std::uintptr_t)ptr,
                 match([](auto &tag) { return get_memory_tag_name(tag); })(info.tag), info.size,
                 info.alignment, info.allocatorType);
    }
#if 0
#  if ZS_ENABLE_CUDA
    deinitialize_backend(exec_cuda);
#  endif
#  if ZS_ENABLE_OPENMP
    deinitialize_backend(exec_omp);
#  endif
    deinitialize_backend(exec_seq);
#endif
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

}  // namespace zs
