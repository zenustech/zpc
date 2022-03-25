#include "Resource.h"

#include "zensim/Port.hpp"
#include "zensim/execution/Concurrency.h"
#include "zensim/memory/Allocator.h"
#include "zensim/memory/MemoryResource.h"
#if defined(ZS_ENABLE_CUDA)
#  include "zensim/cuda/Port.hpp"
#  include "zensim/cuda/memory/Allocator.h"
#endif
#if defined(ZS_ENABLE_OPENMP)
#  include "zensim/omp/Port.hpp"
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

  /// owning upstream should specify deleter
  template <bool IsVirtual, typename T>
  template <template <typename Tag> class ResourceT, typename... Args, std::size_t... Is>
  void ZSPmrAllocator<IsVirtual, T>::setOwningUpstream(mem_tags tag, ProcID devid,
                                                       std::tuple<Args &&...> args,
                                                       index_seq<Is...>) {
    match(
        [&](auto t) ->std::enable_if_t<is_memory_source_available(t)> {
          using MemT = RM_CVREF_T(t);
          res = std::make_unique<ResourceT<MemT>>(devid, std::get<Is>(args)...);
          location = MemoryLocation{t.value, devid};
          cloner = [devid, args]() -> std::unique_ptr<resource_type> {
            std::unique_ptr<resource_type> ret{};
            std::apply(
                [&ret](auto &&...ctorArgs) {
                  ret = std::make_unique<ResourceT<decltype(t)>>(FWD(ctorArgs)...);
                },
                std::tuple_cat(std::make_tuple(devid), args));
            return ret;
          };
        },
        #if 0
        [&](host_mem_tag t) {
          using MemT = decltype(t);
          res = std::make_unique<ResourceT<MemT>>(devid, std::get<Is>(args)...);
          location = MemoryLocation{t.value, devid};
          cloner = [devid, args]() -> std::unique_ptr<resource_type> {
            std::unique_ptr<resource_type> ret{};
            std::apply(
                [&ret](auto &&...ctorArgs) {
                  ret = std::make_unique<ResourceT<decltype(t)>>(FWD(ctorArgs)...);
                },
                std::tuple_cat(std::make_tuple(devid), args));
            return ret;
          };
        },
#if defined(ZS_ENABLE_CUDA)
        [&](auto t)
            -> std::enable_if_t<(
                is_same_v<decltype(t), device_mem_tag> || is_same_v<decltype(t), um_mem_tag>)> {
          using MemT = decltype(t);
          res = std::make_unique<ResourceT<MemT>>(devid, std::get<Is>(args)...);
          location = MemoryLocation{t.value, devid};
          cloner = [devid, args]() -> std::unique_ptr<resource_type> {
            std::unique_ptr<resource_type> ret{};
            std::apply(
                [&ret](auto &&...ctorArgs) {
                  ret = std::make_unique<ResourceT<decltype(t)>>(FWD(ctorArgs)...);
                },
                std::tuple_cat(std::make_tuple(devid), args));
            return ret;
          };
        },
#endif
            #endif
        [](...) {})(tag);
  }
  template <bool IsVirtual, typename T>
  template <template <typename Tag> class ResourceT, typename MemTag, typename... Args>
  void ZSPmrAllocator<IsVirtual, T>::setOwningUpstream(MemTag tag, ProcID devid, Args &&...args) {
    if constexpr (is_same_v<MemTag, mem_tags>)
      setOwningUpstream<ResourceT>(tag, devid, std::forward_as_tuple(FWD(args)...),
                                   std::index_sequence_for<Args...>{});
    else {
#if 0
#if defined(ZS_ENABLE_CUDA)
      if (is_same_v<MemTag, device_mem_tag> || is_same_v<MemTag, um_mem_tag>) {
        res = std::make_unique<ResourceT<MemTag>>(devid, FWD(args)...);
        location = MemoryLocation{MemTag::value, devid};
        cloner = [devid, args = std::make_tuple(args...)]() -> std::unique_ptr<resource_type> {
          std::unique_ptr<resource_type> ret{};
          std::apply(
              [&ret](auto &&...ctorArgs) {
                ret = std::make_unique<ResourceT<MemTag>>(FWD(ctorArgs)...);
              },
              std::tuple_cat(std::make_tuple(devid), args));
          return ret;
        };
      }
#endif
      if (is_same_v<MemTag, host_mem_tag>) {
        res = std::make_unique<ResourceT<MemTag>>(devid, FWD(args)...);
        location = MemoryLocation{MemTag::value, devid};
        cloner = [devid, args = std::make_tuple(args...)]() -> std::unique_ptr<resource_type> {
          std::unique_ptr<resource_type> ret{};
          std::apply(
              [&ret](auto &&...ctorArgs) {
                ret = std::make_unique<ResourceT<MemTag>>(FWD(ctorArgs)...);
              },
              std::tuple_cat(std::make_tuple(devid), args));
          return ret;
        };
      }
      #endif
      if (is_memory_source_available(tag)) {
        res = std::make_unique<ResourceT<MemTag>>(devid, FWD(args)...);
        location = MemoryLocation{MemTag::value, devid};
        cloner = [devid, args = std::make_tuple(args...)]() -> std::unique_ptr<resource_type> {
          std::unique_ptr<resource_type> ret{};
          std::apply(
              [&ret](auto &&...ctorArgs) {
                ret = std::make_unique<ResourceT<MemTag>>(FWD(ctorArgs)...);
              },
              std::tuple_cat(std::make_tuple(devid), args));
          return ret;
        };
      }
    }
  }

  ZSPmrAllocator<> get_memory_source(memsrc_e mre, ProcID devid, std::string_view advice) {
    const mem_tags tag = to_memory_source_tag(mre);
    ZSPmrAllocator<> ret{};
    if (advice.empty()) {
      if (mre == memsrc_e::um) {
        if (devid < -1)
          match([&ret, devid](auto tag) -> std::enable_if_t<is_memory_source_available(tag)
                                                          || is_same_v<RM_CVREF_T(tag), mem_tags>> {
            ret.setOwningUpstream<advisor_memory_resource>(tag, devid, "READ_MOSTLY");
              },
              [](...) {})(tag);
        else
          match(
              [&ret, devid](auto tag) -> std::enable_if_t<is_memory_source_available(tag)
                                                          || is_same_v<RM_CVREF_T(tag), mem_tags>> {
            ret.setOwningUpstream<advisor_memory_resource>(tag, devid, "PREFERRED_LOCATION");
              },
              [](...) {})(tag);
      } else {
        // match([&ret](auto &tag) { ret.setNonOwningUpstream<raw_memory_resource>(tag); })(tag);
        match(
            [&ret, devid](auto tag) -> std::enable_if_t<is_memory_source_available(tag)
                                                        || is_same_v<RM_CVREF_T(tag), mem_tags>> {
          ret.setOwningUpstream<default_memory_resource>(tag, devid);
            },
            [](...) {})(tag);
        // ret.setNonOwningUpstream<raw_memory_resource>(tag);
      }
    } else
      match([&ret, &advice,
             devid](auto tag) -> std::enable_if_t<is_memory_source_available(tag)
                                                   || is_same_v<RM_CVREF_T(tag), mem_tags>> {
        ret.setOwningUpstream<advisor_memory_resource>(tag, devid, advice);
          },
          [](...) {})(tag);
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
          using MemTag = decltype(tag);
#if ZS_ENABLE_CUDA
          if constexpr (is_same_v<MemTag, device_mem_tag>) {
            if (option == "ARENA")
              ret.setOwningUpstream<arena_virtual_memory_resource>(tag, devid, bytes);
            else if (option == "STACK" || option.empty())
              ret.setOwningUpstream<stack_virtual_memory_resource>(tag, devid, bytes);
            else
              throw std::runtime_error(fmt::format("unkonwn vmr option [{}]\n", option));
          }
#endif
          if constexpr (is_same_v<MemTag, host_mem_tag>) {
            if (option == "ARENA")
              ret.setOwningUpstream<arena_virtual_memory_resource>(tag, devid, bytes);
            else if (option == "STACK" || option.empty())
              ret.setOwningUpstream<stack_virtual_memory_resource>(tag, devid, bytes);
            else
              throw std::runtime_error(fmt::format("unkonwn vmr option [{}]\n", option));
          }
        },
        [](...) {})(tag);
    return ret;
  }

  Resource::Resource() {
    #if 0
    initialize_backend(exec_seq);
#if defined(ZS_ENABLE_CUDA)
    initialize_backend(exec_cuda);
#endif
#if defined(ZS_ENABLE_OPENMP)
    initialize_backend(exec_omp);
#endif
    // sycl...
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
    deinitialize_backend(exec_seq);
#if defined(ZS_ENABLE_CUDA)
    deinitialize_backend(exec_cuda);
#endif
#if defined(ZS_ENABLE_OPENMP)
    deinitialize_backend(exec_omp);
#endif
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

  ZPC_API Resource &Resource::instance() noexcept { return s_resource; }

  ZPC_API Resource Resource::s_resource{};

  Resource &get_resource_manager() noexcept { return Resource::instance(); }

}  // namespace zs
