#pragma once

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "zensim/Reflection.h"
#include "zensim/ZpcFunction.hpp"
#include "zensim/memory/MemOps.hpp"
#include "zensim/memory/MemoryResource.h"
// #include "zensim/types/Pointers.hpp"
#include "zensim/memory/Allocator.h"
#include "zensim/types/SmallVector.hpp"
#include "zensim/types/Tuple.h"
#if ZS_ENABLE_CUDA
#  include "zensim/cuda/memory/Allocator.h"
#elif ZS_ENABLE_MUSA
#  include "zensim/musa/memory/Allocator.h"
#elif ZS_ENABLE_ROCM
#  include "zensim/rocm/memory/Allocator.h"
#elif ZS_ENABLE_SYCL
#  include "zensim/sycl/memory/Allocator.h"
#endif
#if ZS_ENABLE_OPENMP
#endif

namespace zs {

  template <bool is_virtual_ = false, typename T = byte> struct ZPC_API ZSPmrAllocator {
    using value_type = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    /// this is different from std::polymorphic_allocator
    using propagate_on_container_move_assignment = true_type;
    using propagate_on_container_copy_assignment = true_type;
    using propagate_on_container_swap = true_type;
    using is_virtual = wrapv<is_virtual_>;
    using resource_type = conditional_t<is_virtual::value, vmr_t, mr_t>;

    ZSPmrAllocator() = default;
    ZSPmrAllocator(ZSPmrAllocator &&) = default;
    ZSPmrAllocator &operator=(ZSPmrAllocator &&) = default;
    ZSPmrAllocator(const ZSPmrAllocator &o) { *this = o.select_on_container_copy_construction(); }
    ZSPmrAllocator &operator=(const ZSPmrAllocator &o) {
      *this = o.select_on_container_copy_construction();
      return *this;
    }

    friend void swap(ZSPmrAllocator &a, ZSPmrAllocator &b) noexcept {
      std::swap(a.res, b.res);
      std::swap(a.location, b.location);
    }

    constexpr resource_type *resource() noexcept { return res.get(); }
    [[nodiscard]] void *allocate(size_t bytes, size_t alignment = alignof(std::max_align_t)) {
      return res->allocate(bytes, alignment);
    }
    void deallocate(void *p, size_t bytes, size_t alignment = alignof(std::max_align_t)) {
      res->deallocate(p, bytes, alignment);
    }
    bool is_equal(const ZSPmrAllocator &other) const noexcept {
      return res.get() == other.res.get() && location == other.location;
    }
    template <bool V = is_virtual::value>
    enable_if_type<V, bool> commit(size_t offset,
                                   size_t bytes = resource_type::s_chunk_granularity) {
      return res->commit(offset, bytes);
    }
    template <bool V = is_virtual::value>
    enable_if_type<V, bool> evict(size_t offset,
                                  size_t bytes = resource_type::s_chunk_granularity) {
      return res->evict(offset, bytes);
    }
    template <bool V = is_virtual::value> enable_if_type<V, bool> check_residency(
        size_t offset, size_t bytes = resource_type::s_chunk_granularity) const {
      return res->check_residency(offset, bytes);
    }
    template <bool V = is_virtual::value>
    enable_if_type<V, void *> address(size_t offset = 0) const {
      return res->address(offset);
    }

    ZSPmrAllocator select_on_container_copy_construction() const {
      ZSPmrAllocator ret{};
      ret.cloner = this->cloner;
      ret.res = this->cloner();
      ret.location = this->location;
      return ret;
    }

    /// owning upstream should specify deleter
    template <template <typename Tag> class ResourceT, typename... Args, size_t... Is>
    void setOwningUpstream(mem_tags tag, ProcID devid, zs::tuple<Args...> args,
                           index_sequence<Is...>) {
      match([&](auto t) {
        if constexpr (is_memory_source_available(t)) {
          using MemT = RM_CVREF_T(t);
          res = std::make_unique<ResourceT<MemT>>(devid, zs::get<Is>(args)...);
          location = MemoryLocation{t.value, devid};
          cloner = [devid, args]() -> std::unique_ptr<resource_type> {
            std::unique_ptr<resource_type> ret{};
            zs::apply(
                [&ret](auto &&...ctorArgs) {
                  ret = std::make_unique<ResourceT<decltype(t)>>(FWD(ctorArgs)...);
                },
                zs::tuple_cat(zs::make_tuple(devid), args));
            return ret;
          };
        } else
          std::cerr << fmt::format("memory resource \"{}\" not available", get_var_type_str(t))
                    << std::endl;
      })(tag);
    }
    template <template <typename Tag> class ResourceT, typename MemTag, typename... Args>
    void setOwningUpstream(MemTag tag, ProcID devid, Args &&...args) {
      if constexpr (is_same_v<MemTag, mem_tags>)
        setOwningUpstream<ResourceT>(tag, devid, zs::forward_as_tuple(FWD(args)...),
                                     index_sequence_for<Args...>{});
      else {
        if constexpr (is_memory_source_available(tag)) {
          res = std::make_unique<ResourceT<MemTag>>(devid, args...);
          location = MemoryLocation{MemTag::value, devid};
          cloner
              = [devid, args = zs::make_tuple(FWD(args)...)]() -> std::unique_ptr<resource_type> {
            std::unique_ptr<resource_type> ret{};
            zs::apply(
                [&ret](auto &&...ctorArgs) {
                  ret = std::make_unique<ResourceT<MemTag>>(FWD(ctorArgs)...);
                },
                zs::tuple_cat(zs::make_tuple(devid), args));
            return ret;
          };
        } else
          std::cerr << fmt::format("memory resource \"{}\" not available", get_var_type_str(tag))
                    << std::endl;
      }
    }

    function<std::unique_ptr<resource_type>()> cloner{};
    std::unique_ptr<resource_type> res{};
    MemoryLocation location{memsrc_e::host, -1};
  };

  extern template struct ZPC_TEMPLATE_IMPORT ZSPmrAllocator<false, byte>;
  extern template struct ZPC_TEMPLATE_IMPORT ZSPmrAllocator<true, byte>;

  /// @note might not be sufficient for multi-GPU-context scenarios
  template <typename Policy, bool is_virtual, typename T>
  bool valid_memspace_for_execution(const Policy &pol,
                                    const ZSPmrAllocator<is_virtual, T> &allocator) {
    constexpr execspace_e space = Policy::exec_tag::value;
#if ZS_ENABLE_CUDA
    if constexpr (space == execspace_e::cuda)
      return allocator.location.memspace() == memsrc_e::device
             || allocator.location.memspace() == memsrc_e::um;
#elif ZS_ENABLE_MUSA
    if constexpr (space == execspace_e::musa)
      return allocator.location.memspace() == memsrc_e::device
             || allocator.location.memspace() == memsrc_e::um;
#elif ZS_ENABLE_ROCM
    if constexpr (space == execspace_e::rocm)
      return allocator.location.memspace() == memsrc_e::device
             || allocator.location.memspace() == memsrc_e::um;
#elif ZS_ENABLE_SYCL
    if constexpr (space == execspace_e::sycl)
      return allocator.location.memspace() == memsrc_e::device
             || allocator.location.memspace() == memsrc_e::um;
#endif
#if ZS_ENABLE_OPENMP
    if constexpr (space == execspace_e::openmp)
      return allocator.location.memspace() == memsrc_e::host;
#endif
    /// @note sequential (host)
    return allocator.location.memspace() == memsrc_e::host;
  }

  template <typename Allocator> struct is_zs_allocator : false_type {};
  template <bool is_virtual, typename T> struct is_zs_allocator<ZSPmrAllocator<is_virtual, T>>
      : true_type {};
  template <typename Allocator> using is_virtual_zs_allocator
      = conditional_t<is_zs_allocator<Allocator>::value, typename Allocator::is_virtual,
                      false_type>;

  template <typename MemTag> constexpr bool is_memory_source_available(MemTag) noexcept {
    if constexpr (is_same_v<MemTag, device_mem_tag>)
      return ZS_ENABLE_DEVICE;
    else if constexpr (is_same_v<MemTag, um_mem_tag>)
      return ZS_ENABLE_DEVICE;
    else if constexpr (is_same_v<MemTag, host_mem_tag>)
      return true;
    return false;
  }

  inline ZPC_API ZSPmrAllocator<> get_memory_source(memsrc_e mre, ProcID devid,
                                                    std::string_view advice = std::string_view{}) {
    const mem_tags tag = to_memory_source_tag(mre);
    ZSPmrAllocator<> ret{};
    if (advice.empty()) {
      if (mre == memsrc_e::um) {
        if (devid < -1)
          match([&ret, devid](auto tag) {
            if constexpr (is_memory_source_available(tag) || is_same_v<RM_CVREF_T(tag), mem_tags>)
              ret.setOwningUpstream<advisor_memory_resource>(tag, devid, "READ_MOSTLY");
            else
              std::cerr << fmt::format(
                  "invalid allocations of memory resource \"{}\" with advice \"READ_MOSTLY\"",
                  get_var_type_str(tag))
                        << std::endl;
          })(tag);
        else
          match([&ret, devid](auto tag) {
            if constexpr (is_memory_source_available(tag) || is_same_v<RM_CVREF_T(tag), mem_tags>)
              ret.setOwningUpstream<advisor_memory_resource>(tag, devid, "PREFERRED_LOCATION");
            else
              std::cerr << fmt::format(
                  "invalid allocations of memory resource \"{}\" with advice "
                  "\"PREFERRED_LOCATION\"",
                  get_var_type_str(tag))
                        << std::endl;
          })(tag);
      } else {
        // match([&ret](auto &tag) { ret.setNonOwningUpstream<raw_memory_resource>(tag); })(tag);
        match([&ret, devid](auto tag) {
          if constexpr (is_memory_source_available(tag) || is_same_v<RM_CVREF_T(tag), mem_tags>)
            ret.setOwningUpstream<default_memory_resource>(tag, devid);
          else
            std::cerr << fmt::format("invalid default allocations of memory resource \"{}\"",
                                     get_var_type_str(tag))
                      << std::endl;
        })(tag);
        // ret.setNonOwningUpstream<raw_memory_resource>(tag);
      }
    } else
      match([&ret, &advice, devid](auto tag) {
        if constexpr (is_memory_source_available(tag) || is_same_v<RM_CVREF_T(tag), mem_tags>)
          ret.setOwningUpstream<advisor_memory_resource>(tag, devid, advice);
        else
          std::cerr << fmt::format(
              "invalid advice \"{}\" for allocations of memory resource \"{}\"", advice,
              get_var_type_str(tag))
                    << std::endl;
      })(tag);
    return ret;
  }

  inline ZPC_API ZSPmrAllocator<true> get_virtual_memory_source(memsrc_e mre, ProcID devid,
                                                                size_t bytes,
                                                                std::string_view option = "STACK") {
    const mem_tags tag = to_memory_source_tag(mre);
    ZSPmrAllocator<true> ret{};
    if (mre == memsrc_e::um)
      throw std::runtime_error("no corresponding virtual memory resource for [um]");
    match([&ret, devid, bytes, option](auto tag) {
      if constexpr (!is_same_v<decltype(tag), um_mem_tag>)
        if constexpr (is_memory_source_available(tag)) {
          if (option == "ARENA")
            ret.setOwningUpstream<arena_virtual_memory_resource>(tag, devid, bytes);
          else if (option == "STACK" || option.empty())
            ret.setOwningUpstream<stack_virtual_memory_resource>(tag, devid, bytes);
          else
            throw std::runtime_error(fmt::format("unkonwn vmr option [{}]\n", option));
          return;
        }
      std::cerr << fmt::format(
          "invalid option \"{}\" for allocations of virtual memory resource \"{}\".", option,
          get_var_type_str(tag).asChars())
                << std::endl;
    })(tag);
    return ret;
  }

  template <execspace_e space> constexpr bool initialize_backend(wrapv<space>) noexcept {
    return false;
  }

  struct ZPC_API Resource {
    static std::atomic_ullong &counter() noexcept;
    static Resource &instance() noexcept;
    static void copy(MemoryEntity dst, MemoryEntity src, size_t numBytes) {
      if (dst.location.onHost() && src.location.onHost())
        zs::copy(mem_host, dst.ptr, src.ptr, numBytes);
      else {
        if constexpr (is_memory_source_available(mem_device)) {
          if (!dst.location.onHost() && !src.location.onHost())
            zs::copyDtoD(mem_device, dst.ptr, src.ptr, numBytes);
          else if (dst.location.onHost() && !src.location.onHost())
            zs::copyDtoH(mem_device, dst.ptr, src.ptr, numBytes);
          else if (!dst.location.onHost() && src.location.onHost())
            zs::copyHtoD(mem_device, dst.ptr, src.ptr, numBytes);
        } else
          throw std::runtime_error("There is no corresponding device backend for Resource::copy");
      }
    }
    static void memset(MemoryEntity dst, char ch, size_t numBytes) {
      if (dst.location.onHost())
        zs::memset(mem_host, dst.ptr, ch, numBytes);
      else {
        if constexpr (is_memory_source_available(mem_device))
          zs::memset(mem_device, dst.ptr, ch, numBytes);
        else
          throw std::runtime_error("There is no corresponding device backend for Resource::memset");
      }
    }

    struct AllocationRecord {
      mem_tags tag{};
      size_t size{0}, alignment{0};
      std::string allocatorType{};
    };
    Resource();
    ~Resource();

    void record(mem_tags tag, void *ptr, std::string_view name, size_t size, size_t alignment);
    void erase(void *ptr);

    void deallocate(void *ptr);

  private:
    mutable std::atomic_ullong _counter{0};
  };

  inline auto select_properties(const std::vector<PropertyTag> &props,
                                const std::vector<SmallString> &names) {
    std::vector<PropertyTag> ret(0);
    for (auto &&name : names)
      for (auto &&prop : props)
        if (prop.name == name) {
          ret.push_back(prop);
          break;
        }
    return ret;
  }

}  // namespace zs
