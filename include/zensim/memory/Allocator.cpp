#include "Allocator.h"

#include "zensim/Logger.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/core.h"
#include "zensim/zpc_tpls/fmt/format.h"

#if defined(ZS_PLATFORM_UNIX)
#  include <sys/mman.h>
#  include <unistd.h>
#elif defined(ZS_PLATFORM_WINDOWS)
#  define NOMINMAX
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

namespace zs {

  template struct ZPC_CORE_TEMPLATE_EXPORT advisor_memory_resource<host_mem_tag>;
#if defined(ZS_PLATFORM_UNIX)

#  if 0
  stack_virtual_memory_resource<host_mem_tag>::stack_virtual_memory_resource(ProcID did,
                                                                             std::string_view type)
      : _type{type},
        _granularity{0},
        _addr{nullptr},
        _offset{0},
        _allocatedSpace{0},
        _reservedSpace{0},
        _did{did} {
    if (did >= 0)
      throw std::runtime_error(
          fmt::format("hostvm target device index [{}] is not negative", (int)did));
    if (type != "HOST_VIRTUAL")
      throw std::runtime_error(
          fmt::format("currently hostvm does not support allocation type {}", type));

    _granularity = (size_t)getpagesize();
  }

  stack_virtual_memory_resource<host_mem_tag>::~stack_virtual_memory_resource() {
    if (_reservedSpace) munmap(_addr, _reservedSpace);
  }

  bool stack_virtual_memory_resource<host_mem_tag>::reserve(size_t desiredSpace) {
    if (desiredSpace <= _reservedSpace) return true;

    auto newSpace = (desiredSpace + _granularity - 1) / _granularity * _granularity;
    if (_reservedSpace == 0) {
      auto ret = mmap(nullptr, newSpace, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, 0, 0);
      if (ret == MAP_FAILED) return false;
      _addr = ret;
      _reservedSpace = newSpace;
      return true;
    }
    auto ret = mremap(_addr, _reservedSpace, newSpace, MREMAP_MAYMOVE);
    if (ret == MAP_FAILED) return false;
    if (_addr != ret) _addr = ret;
    _reservedSpace = newSpace;
    return true;
  }

  void *stack_virtual_memory_resource<host_mem_tag>::do_allocate(size_t bytes,
                                                                 size_t alignment) {
    _offset = (_offset + alignment - 1) / alignment * alignment;

    if (!reserve(_offset + bytes)) return nullptr;

    if (_offset + bytes <= _allocatedSpace) {
      void *ret = (void *)((char *)_addr + _offset);
      _offset += bytes;
      return ret;
    }

    auto allocationBytes = (bytes + _granularity - 1) / _granularity * _granularity;
    if (mprotect((char *)_addr + _allocatedSpace, allocationBytes, PROT_READ | PROT_WRITE) == 0) {
      _allocatedSpace += allocationBytes;
      auto ret = (char *)_addr + _offset;
      _offset += bytes;
      return (void *)ret;
    }
    return nullptr;
  }

  void stack_virtual_memory_resource<host_mem_tag>::do_deallocate(void *ptr, size_t bytes,
                                                                  size_t alignment) {
    auto split = ((size_t)ptr + _granularity - 1) / _granularity * _granularity;
    if (split < (size_t)_addr) split = (size_t)_addr;
    bytes = (split - (size_t)_addr);
    if (bytes > _allocatedSpace) return;
    bytes = (size_t)_allocatedSpace - bytes;
    if (madvise((void *)split, bytes, MADV_DONTNEED) != 0) return;
    if (mprotect((void *)split, bytes, PROT_NONE) == 0) {
      _allocatedSpace -= bytes;
      if (_allocatedSpace < _offset) _offset = _allocatedSpace;
    }
  }
#  endif  // disable this stack vmr impl

  arena_virtual_memory_resource<host_mem_tag>::arena_virtual_memory_resource(ProcID did,
                                                                             size_t space)
      : _did{did}, _reservedSpace{round_up(space, s_chunk_granularity)} {
    if (did >= 0)
      throw std::runtime_error(
          fmt::format("hostvm target device index [{}] is not negative", (int)did));
    _granularity = (size_t)getpagesize();
    auto ret = mmap(nullptr, _reservedSpace, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, 0, 0);
    if (ret == MAP_FAILED)
      throw std::runtime_error(
          fmt::format("failed to reserve a virtual address range of size {}", _reservedSpace));
    _addr = ret;
    _activeChunkMasks.resize((_reservedSpace / s_chunk_granularity + 63) / 64, (u64)0);
  }

  arena_virtual_memory_resource<host_mem_tag>::~arena_virtual_memory_resource() {
    munmap(_addr, _reservedSpace);
  }

  bool arena_virtual_memory_resource<host_mem_tag>::do_check_residency(size_t offset,
                                                                       size_t bytes) const {
    size_t st = round_down(offset, s_chunk_granularity);
    if (st >= _reservedSpace) return false;
    offset += bytes;
    size_t ed = offset <= _reservedSpace ? round_up(offset, s_chunk_granularity) : _reservedSpace;
    for (st >>= s_chunk_granularity_bits, ed >>= s_chunk_granularity_bits; st != ed; ++st)
      if ((_activeChunkMasks[st >> 6] & ((u64)1 << (st & 63))) == 0) return false;
    return true;
  }
  bool arena_virtual_memory_resource<host_mem_tag>::do_commit(size_t offset, size_t bytes) {
    size_t st = round_down(offset, s_chunk_granularity);
    if (st >= _reservedSpace) return false;
    offset += bytes;
    size_t ed = offset <= _reservedSpace ? round_up(offset, s_chunk_granularity) : _reservedSpace;

    if (mprotect((char *)_addr + st, ed - st, PROT_READ | PROT_WRITE) == 0) {
      for (st >>= s_chunk_granularity_bits, ed >>= s_chunk_granularity_bits; st != ed; ++st)
        _activeChunkMasks[st >> 6] |= ((u64)1 << (st & 63));
      return true;
    }
    return false;
  }

  bool arena_virtual_memory_resource<host_mem_tag>::do_evict(size_t offset, size_t bytes) {
    size_t st = round_up(offset, s_chunk_granularity);
    offset += bytes;
    size_t ed = offset <= _reservedSpace ? round_down(offset, s_chunk_granularity) : _reservedSpace;
    if (st >= ed) return false;
    bytes = ed - st;
    if (madvise((void *)((char *)_addr + st), bytes, MADV_DONTNEED) != 0) return false;
    if (mprotect((void *)((char *)_addr + st), bytes, PROT_NONE) != 0) return false;
    for (st >>= s_chunk_granularity_bits, ed >>= s_chunk_granularity_bits; st != ed; ++st)
      _activeChunkMasks[st >> 6] &= ~((u64)1 << (st & 63));
    return true;
  }
#endif  // end UNIX

  stack_virtual_memory_resource<host_mem_tag>::stack_virtual_memory_resource(ProcID did,
                                                                             size_t size)
      : _allocatedSpace{0}, _did{did} {
    if (did >= 0)
      throw std::runtime_error(
          fmt::format("hostvm target device index [{}] is not negative", (int)did));

#if defined(ZS_PLATFORM_WINDOWS)
    {
      auto info = SYSTEM_INFO{};
      GetSystemInfo(&info);
      _granularity = (size_t)info.dwAllocationGranularity;
    }
#elif defined(ZS_PLATFORM_UNIX)
    _granularity = (size_t)getpagesize();
#endif
    if (s_chunk_granularity % _granularity != 0)
      throw std::runtime_error("chunk granularity not a multiple of alloc granularity.");

    _reservedSpace = round_up(size, s_chunk_granularity);

#if defined(ZS_PLATFORM_WINDOWS)
    _addr = VirtualAlloc(nullptr, _reservedSpace, MEM_RESERVE, PAGE_NOACCESS);
    if (_addr == nullptr) {
      auto const err = GetLastError();
      throw std::system_error(std::error_code(err, std::system_category()),
                              "fails to reserve a host virtual range.");
    }
#elif defined(ZS_PLATFORM_UNIX)
    _addr = mmap(nullptr, _reservedSpace, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, 0, 0);
    if (_addr == MAP_FAILED) throw std::runtime_error("fails to reserve a host virtual range.");
#endif
  }

  stack_virtual_memory_resource<host_mem_tag>::~stack_virtual_memory_resource() {
    if (_reservedSpace) {
#if defined(ZS_PLATFORM_WINDOWS)
      (void)VirtualFree(_addr, 0, MEM_RELEASE);
#elif defined(ZS_PLATFORM_UNIX)
      (void)munmap(_addr, _reservedSpace);
#endif
    }
  }

  bool stack_virtual_memory_resource<host_mem_tag>::do_check_residency(size_t offset,
                                                                       size_t bytes) const {
    offset += bytes;
    size_t ed = offset <= _reservedSpace ? round_up(offset, s_chunk_granularity) : _reservedSpace;
    return ed <= _allocatedSpace;
  }
  bool stack_virtual_memory_resource<host_mem_tag>::do_commit(size_t offset, size_t bytes) {
    offset += bytes;
    size_t ed = offset <= _reservedSpace ? round_up(offset, s_chunk_granularity) : _reservedSpace;
    if (ed < _allocatedSpace) return true;

#if defined(ZS_PLATFORM_WINDOWS)
    if (VirtualAlloc(_addr, ed, MEM_COMMIT, PAGE_READWRITE) == nullptr) return false;
#elif defined(ZS_PLATFORM_UNIX)
    if (mprotect(_addr, ed, PROT_READ | PROT_WRITE) != 0) return false;
#endif

    _allocatedSpace = ed;
    return true;
  }

  bool stack_virtual_memory_resource<host_mem_tag>::do_evict(size_t offset, size_t bytes) {
    ZS_WARN_IF(round_down(offset + bytes, s_chunk_granularity) < _allocatedSpace,
               "will evict more bytes (till the end) than asking");
    size_t st = round_up(offset, s_chunk_granularity);
    if (st >= _allocatedSpace) return true;

#if defined(ZS_PLATFORM_WINDOWS)
    if (VirtualFree((void *)((char *)_addr + st), _allocatedSpace - st, MEM_DECOMMIT) == 0)
      return false;
#elif defined(ZS_PLATFORM_UNIX)
    if (madvise((void *)((char *)_addr + st), _allocatedSpace - st, MADV_DONTNEED) != 0)
      return false;
    if (mprotect((void *)((char *)_addr + st), _allocatedSpace - st, PROT_NONE) != 0) return false;
#endif

    _allocatedSpace = st;
    return true;
  }

  void *stack_virtual_memory_resource<host_mem_tag>::do_allocate(size_t bytes, size_t alignment) {
    return nullptr;
  }

  void stack_virtual_memory_resource<host_mem_tag>::do_deallocate(void *ptr, size_t bytes,
                                                                  size_t alignment) {}

  /// handle_resource
  handle_resource::handle_resource(mr_t *upstream) noexcept : _upstream{upstream} {}
  handle_resource::handle_resource(size_t initSize, mr_t *upstream) noexcept
      : _bufSize{initSize}, _upstream{upstream} {}
  handle_resource::handle_resource() noexcept
      : handle_resource{&raw_memory_resource<host_mem_tag>::instance()} {}
  handle_resource::~handle_resource() {
    _upstream->deallocate(_handle, _bufSize, _align);
    fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::dark_sea_green),
               "deallocate {} bytes in handle_resource\n", _bufSize);
  }

  void *handle_resource::do_allocate(size_t bytes, size_t alignment) {
    if (_handle == nullptr) {
      _handle = _head = (char *)(_upstream->allocate(_bufSize, alignment));
      _align = alignment;
      fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::dark_sea_green),
                 "initially allocate {} bytes in handle_resource\n", _bufSize);
    }
    char *ret = _head + alignment - 1 - ((size_t)_head + alignment - 1) % alignment;
    _head = ret + bytes;
    if (_head < _handle + _bufSize) return ret;

    _upstream->deallocate(_handle, _bufSize, alignment);

    auto offset = ret - _handle;  ///< ret is offset
    _bufSize = (size_t)(_head - _handle) << 1;
    _align = alignment;
    _handle = (char *)(_upstream->allocate(_bufSize, alignment));
    fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::dark_sea_green),
               "reallocate {} bytes in handle_resource\n", _bufSize);
    ret = _handle + offset;
    _head = ret + bytes;
    return ret;
  }
  void handle_resource::do_deallocate(void *p, size_t bytes, size_t alignment) {
    if (p >= _head)
      throw std::bad_alloc{};
    else if (p < _handle)
      throw std::bad_alloc{};
    _head = (char *)p;
  }
  bool handle_resource::do_is_equal(const mr_t &other) const noexcept {
    return this == dynamic_cast<handle_resource *>(const_cast<mr_t *>(&other));
  }

  /// stack allocator
  stack_allocator::stack_allocator(mr_t *mr, size_t totalMemBytes, size_t alignBytes)
      : _align{alignBytes}, _mr{mr} {
    _data = _head = (char *)(_mr->allocate(totalMemBytes, _align));
    _tail = _head + totalMemBytes;  ///< not so sure about this
  };
  stack_allocator::~stack_allocator() {
    _mr->deallocate((void *)_data, (size_t)(_tail - _data), _align);
  }

  /// from taichi
  void *stack_allocator::allocate(size_t bytes) {
    /// first align head
    char *ret = _head + _align - 1 - ((size_t)_head + _align - 1) % _align;
    _head = ret + bytes;
    if (_head > _tail)
      throw std::bad_alloc{};
    else
      return ret;
  }
  void stack_allocator::deallocate(void *p, size_t) {
    if (p >= _head)
      throw std::bad_alloc{};
    else if (p < _data)
      throw std::bad_alloc{};
    _head = (char *)p;
  }

}  // namespace zs