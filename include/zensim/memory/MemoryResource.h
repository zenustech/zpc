#pragma once
#include <cstddef>
#include <memory>
#include <stdexcept>

#include "zensim/Platform.hpp"
#include "zensim/types/Function.h"
#include "zensim/types/Polymorphism.h"
#include "zensim/types/Property.h"
#include "zensim/zpc_tpls/fmt/format.h"

#ifdef ZS_PLATFORM_WINDOWS
#  include <memory_resource>
#endif
namespace zs {

  // namespace pmr = std::pmr;

  /// since we cannot use memory_resource header in libstdc++ (chrono_nvcc compilation issue)
  /// we directly use its implementation
#ifdef ZS_PLATFORM_WINDOWS

  using mr_t = std::pmr::memory_resource;

#else
  class memory_resource {
    static constexpr size_t _S_max_align = alignof(max_align_t);

  public:
    memory_resource() = default;
    memory_resource(const memory_resource&) = default;
    virtual ~memory_resource() = default;  // key function

    memory_resource& operator=(const memory_resource&) = default;

    [[nodiscard]] void* allocate(size_t __bytes, size_t __alignment = _S_max_align)
        __attribute__((__returns_nonnull__, __alloc_size__(2), __alloc_align__(3))) {
      return do_allocate(__bytes, __alignment);
    }

    void deallocate(void* __p, size_t __bytes, size_t __alignment = _S_max_align)
        __attribute__((__nonnull__)) {
      return do_deallocate(__p, __bytes, __alignment);
    }

    bool is_equal(const memory_resource& __other) const noexcept { return do_is_equal(__other); }

  private:
    virtual void* do_allocate(size_t __bytes, size_t __alignment) = 0;

    virtual void do_deallocate(void* __p, size_t __bytes, size_t __alignment) = 0;

    virtual bool do_is_equal(const memory_resource& __other) const noexcept = 0;
  };

  inline bool operator==(const memory_resource& __a, const memory_resource& __b) noexcept {
    return &__a == &__b || __a.is_equal(__b);
  }

#  if __cpp_impl_three_way_comparison < 201907L
  inline bool operator!=(const memory_resource& __a, const memory_resource& __b) noexcept {
    return !(__a == __b);
  }
#  endif

  using mr_t = memory_resource;
#endif

  struct vmr_t : public mr_t {
    static constexpr size_t s_chunk_granularity_bits = (size_t)21;
    static constexpr size_t s_chunk_granularity = (size_t)1 << s_chunk_granularity_bits;
    vmr_t() = default;
    vmr_t(const vmr_t&) = default;
    vmr_t& operator=(const vmr_t&) = default;
    virtual ~vmr_t() = default;

    bool commit(size_t offset, size_t bytes = s_chunk_granularity) {
      return do_commit(offset, bytes);
    }
    bool evict(size_t offset, size_t bytes = s_chunk_granularity) {
      return do_evict(offset, bytes);
    }
    bool check_residency(size_t offset, size_t bytes = s_chunk_granularity) const {
      return do_check_residency(offset, bytes);
    }
    void* address(size_t offset = 0) const { return do_address(offset); }

  private:
    void* do_allocate(size_t, size_t) override { return nullptr; }
    void do_deallocate(void*, size_t, size_t) override {}
    bool do_is_equal(const mr_t& o) const noexcept override { return this == &o; }
    virtual bool do_commit(size_t, size_t) = 0;
    virtual bool do_evict(size_t, size_t) = 0;
    virtual bool do_check_residency(size_t, size_t) const = 0;
    virtual void* do_address(size_t) const = 0;
  };

  // using unsynchronized_pool_resource = pmr::unsynchronized_pool_resource;
  // using synchronized_pool_resource = pmr::synchronized_pool_resource;
  // template <typename T> using object_allocator = pmr::polymorphic_allocator<T>;

  using mem_tags = variant<host_mem_tag, device_mem_tag, um_mem_tag>;

  constexpr mem_tags to_memory_source_tag(memsrc_e loc) noexcept {
    mem_tags ret{};
    switch (loc) {
      case memsrc_e::host:
        ret = mem_host;
        break;
      case memsrc_e::device:
        ret = mem_device;
        break;
      case memsrc_e::um:
        ret = mem_um;
        break;
      default:;
    }
    return ret;
  }

  constexpr const char* memory_source_tag[] = {"HOST", "DEVICE", "UM"};
  constexpr const char* get_memory_tag_name(memsrc_e loc) {
    return memory_source_tag[static_cast<unsigned char>(loc)];
  }

  constexpr memsrc_e get_memory_tag_enum(const mem_tags& tag) {
    if (std::holds_alternative<host_mem_tag>(tag))
      return memsrc_e::host;
    else if (std::holds_alternative<device_mem_tag>(tag))
      return memsrc_e::device;
    else if (std::holds_alternative<um_mem_tag>(tag))
      return memsrc_e::um;
    return memsrc_e::host;
  }

  struct MemoryTraits {
    /// access mode: read, write or both
    enum : unsigned char { rw = 0, read, write } _access{rw};
    /// data management strategy when accessed by a remote backend: explicit, implicit
    enum : unsigned char { exp = 0, imp } _move{exp};
  };

  struct MemoryLocation {
    constexpr ProcID devid() const noexcept { return _devid; }
    constexpr memsrc_e memspace() const noexcept { return _memsrc; }

    constexpr bool onHost() const noexcept { return _memsrc == memsrc_e::host; }
    constexpr const char* memSpaceName() const { return get_memory_tag_name(memspace()); }
    constexpr mem_tags getTag() const { return to_memory_source_tag(_memsrc); }

    void swap(MemoryLocation& o) noexcept {
      std::swap(_devid, o._devid);
      std::swap(_memsrc, o._memsrc);
    }
    friend void swap(MemoryLocation& a, MemoryLocation& b) noexcept { a.swap(b); }

    friend constexpr bool operator==(const MemoryLocation& a, const MemoryLocation& b) noexcept {
      return a._memsrc == b._memsrc && a._devid == b._devid;
    }
    friend constexpr bool operator!=(const MemoryLocation& a, const MemoryLocation& b) noexcept {
      return !(a == b);
    }

    memsrc_e _memsrc{memsrc_e::host};  // memory source
    ProcID _devid{-1};                 // cpu id
  };

  struct MemoryProperty : MemoryLocation {
    MemoryProperty() = default;
    constexpr explicit MemoryProperty(memsrc_e mre, ProcID devid) : MemoryLocation{mre, devid} {}
    constexpr explicit MemoryProperty(MemoryLocation loc) : MemoryLocation{loc} {}

    constexpr MemoryTraits traits() const noexcept { return _traits; }
    constexpr MemoryProperty memoryProperty() const noexcept {
      return static_cast<MemoryProperty>(*this);
    }

    void swap(MemoryProperty& o) noexcept {
      std::swap(_traits, o._traits);
      std::swap(static_cast<MemoryLocation&>(*this), static_cast<MemoryLocation&>(o));
    }
    friend void swap(MemoryProperty& a, MemoryProperty& b) noexcept { a.swap(b); }

    /// behavior traits
    MemoryTraits _traits{};
  };
  using MemoryHandle = MemoryProperty;

  struct MemoryEntity {
    MemoryLocation location{};
    void* ptr{nullptr};
    MemoryEntity() = default;
    constexpr explicit MemoryEntity(MemoryLocation location, void* ptr)
        : location{location}, ptr{ptr} {}
    constexpr explicit MemoryEntity(MemoryProperty prop, void* ptr) : location{prop}, ptr{ptr} {}
  };

  /// this should be refactored
  // host = 0, device, um
  constexpr mem_tags memop_tag(const MemoryHandle a, const MemoryHandle b) {
    auto spaceA = static_cast<unsigned char>(a.memspace());
    auto spaceB = static_cast<unsigned char>(b.memspace());
    if (spaceA > spaceB) std::swap(spaceA, spaceB);
    if (a.memspace() == b.memspace()) return to_memory_source_tag(a.memspace());
    /// avoid um issue
    else if (spaceB < static_cast<unsigned char>(memsrc_e::um))
      return to_memory_source_tag(memsrc_e::device);
    else if (spaceB == static_cast<unsigned char>(memsrc_e::um))
      return to_memory_source_tag(memsrc_e::um);
    else
      throw std::runtime_error(fmt::format("memop_tag for ({}, {}) is undefined!",
                                           get_memory_tag_name(a.memspace()),
                                           get_memory_tag_name(b.memspace())));
  }

}  // namespace zs
