#include "Resource.h"

#include "zensim/execution/Concurrency.h"
#include "zensim/memory/MemoryResource.h"

namespace zs {

  template struct ZPC_TEMPLATE_EXPORT ZSPmrAllocator<false, byte>;
  template struct ZPC_TEMPLATE_EXPORT ZSPmrAllocator<true, byte>;

  static std::shared_mutex g_resource_rw_mutex{};
  static concurrent_map<void *, Resource::AllocationRecord> g_resource_records;

#if 0
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
#if 0
    initialize_backend(seq_c);
#  if ZS_ENABLE_OPENMP
    puts("openmp initialized");
    initialize_backend(omp_c);
#  endif
#  if ZS_ENABLE_CUDA
    puts("cuda initialized");
    initialize_backend(cuda_c);
#  elif ZS_ENABLE_MUSA
    puts("musa initialized");
    initialize_backend(musa_c);
#  elif ZS_ENABLE_ROCM
    puts("rocm initialized");
    initialize_backend(rocm_c);
#  elif ZS_ENABLE_SYCL
    puts("sycl initialized");
    initialize_backend(sycl_c);
#  endif
#endif
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
    deinitialize_backend(cuda_c);
#  elif ZS_ENABLE_MUSA
    deinitialize_backend(musa_c);
#  elif ZS_ENABLE_ROCM
    deinitialize_backend(rocm_c);
#  elif ZS_ENABLE_SYCL
    deinitialize_backend(sycl_c);
#  endif
#  if ZS_ENABLE_OPENMP
    deinitialize_backend(omp_c);
#  endif
    deinitialize_backend(seq_c);
#endif
  }
  void Resource::record(mem_tags tag, void *ptr, std::string_view name, size_t size,
                        size_t alignment) {
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
