#include "Allocator.h"

#include <zensim/tpls/fmt/color.h>
#include <zensim/tpls/fmt/core.h>
#include <zensim/tpls/fmt/format.h>

#include <zensim/Logger.hpp>

namespace zs {

  /// heap memory resource
  void *heap_memory_source::do_allocate(std::size_t bytes, std::size_t alignment) {
    return aligned_alloc(alignment, bytes);
  }
  void heap_memory_source::do_deallocate(void *p, std::size_t bytes, std::size_t alignment) {
    return free(p);
  }
  bool heap_memory_source::do_is_equal(const mr_t &other) const noexcept {
    return this == dynamic_cast<heap_memory_source *>(const_cast<mr_t *>(&other));
  }

  /// stack memory resource
  void *stack_memory_source::do_allocate(std::size_t bytes, std::size_t alignment) {
    /// maybe variadic length array is appropriate
    return alloca(bytes);
  }
  void stack_memory_source::do_deallocate(void *p, std::size_t bytes, std::size_t alignment) {}
  bool stack_memory_source::do_is_equal(const mr_t &other) const noexcept {
    return this == dynamic_cast<stack_memory_source *>(const_cast<mr_t *>(&other));
  }

  /// handle_resource
  handle_resource::handle_resource(mr_t *upstream) noexcept : _upstream{upstream} {}
  handle_resource::handle_resource(std::size_t initSize, mr_t *upstream) noexcept
      : _bufSize{initSize}, _upstream{upstream} {}
  handle_resource::handle_resource() noexcept : handle_resource{pmr::get_default_resource()} {}
  handle_resource::~handle_resource() {
    _upstream->deallocate(_handle, _bufSize, _align);
    fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::dark_sea_green),
               "deallocate {} bytes in handle_resource\n", _bufSize);
  }

  void *handle_resource::do_allocate(std::size_t bytes, std::size_t alignment) {
    if (_handle == nullptr) {
      _handle = _head = (char *)(_upstream->allocate(_bufSize, alignment));
      _align = alignment;
      fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::dark_sea_green),
                 "initially allocate {} bytes in handle_resource\n", _bufSize);
    }
    char *ret = _head + alignment - 1 - ((std::size_t)_head + alignment - 1) % alignment;
    _head = ret + bytes;
    if (_head < _handle + _bufSize) return ret;

    _upstream->deallocate(_handle, _bufSize, alignment);

    auto offset = ret - _handle;  ///< ret is offset
    _bufSize = (std::size_t)(_head - _handle) << 1;
    _align = alignment;
    _handle = (char *)(_upstream->allocate(_bufSize, alignment));
    fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::dark_sea_green),
               "reallocate {} bytes in handle_resource\n", _bufSize);
    ret = _handle + offset;
    _head = ret + bytes;
    return ret;
  }
  void handle_resource::do_deallocate(void *p, std::size_t bytes, std::size_t alignment) {
    if (p >= _head)
      throw std::bad_alloc{};
    else if (p < _handle)
      throw std::bad_alloc{};
    _head = (char *)p;
  }
  bool handle_resource::do_is_equal(const mr_t &other) const noexcept {
    return this == dynamic_cast<handle_resource *>(const_cast<mr_t *>(&other));
  }

  /// logging resource
  mr_callback display_mr_callback(std::string tag) {
    mr_callback tmp{};
    tmp.alloc = [tag](std::size_t bytes, std::size_t alignment) {
      fmt::print("{}", "[{}]:\tAllocating {} bytes\n", tag, bytes);
    };
    tmp.dealloc = [tag](void *p, std::size_t bytes, std::size_t alignment) {
      fmt::print("{}", "[{}]:\tDeallocating {} bytes\n", tag, bytes);
    };
    tmp.is_equal = [](const mr_t &o) {};
    return tmp;
  }

  mr_callback logging_mr_callback(std::string tag) {
    mr_callback tmp{};
    tmp.alloc = [tag](std::size_t bytes, std::size_t alignment) {
      ZS_TRACE(fmt::format("[{}]:\tAllocating {} bytes\n", tag, bytes));
    };
    tmp.dealloc = [tag](void *p, std::size_t bytes, std::size_t alignment) {
      ZS_TRACE(fmt::format("[{}]:\tDeallocating {} bytes\n", tag, bytes));
    };
    tmp.is_equal = [](const mr_t &o) {};
    return tmp;
  }

  /// stack allocator
  stack_allocator::stack_allocator(mr_t *mr, std::size_t totalMemBytes, std::size_t alignBytes)
      : _mr{mr}, _align{alignBytes} {
    _data = _head = (char *)(_mr->allocate(totalMemBytes, _align));
    _tail = _head + totalMemBytes;  ///< not so sure about this
  };
  stack_allocator::~stack_allocator() {
    _mr->deallocate((void *)_data, (std::size_t)(_tail - _data), _align);
  }

  /// from taichi
  void *stack_allocator::allocate(std::size_t bytes) {
    /// first align head
    char *ret = _head + _align - 1 - ((std::size_t)_head + _align - 1) % _align;
    _head = ret + bytes;
    if (_head > _tail)
      throw std::bad_alloc{};
    else
      return ret;
  }
  void stack_allocator::deallocate(void *p, std::size_t) {
    if (p >= _head)
      throw std::bad_alloc{};
    else if (p < _data)
      throw std::bad_alloc{};
    _head = (char *)p;
  }

}  // namespace zs