#pragma once
#include "zensim/memory/MemoryResource.h"
#include "zensim/execution/ExecutionPolicy.hpp"

namespace zs {

template <execspace_e exec, typename = void>
struct mem_copy {
    void operator()(MemoryEntity dst, MemoryEntity src, std::size_t size) const {
        throw std::runtime_error(fmt::format("copy operation backend {} for [{}, {}, {}] -> [{}, {}, {}] not implemented\n", get_execution_space_tag(exec), src.descr.memSpaceName(), (int)src.descr.devid(), (std::uintptr_t)src.ptr, dst.descr.memSpaceName(), (int)dst.descr.devid(), (std::uintptr_t)dst.ptr));
    }
};

template <execspace_e exec>
struct mem_copy<exec, void_t<std::enable_if_t<exec == execspace_e::host || exec == execspace_e::openmp>>> {
    void operator()(MemoryEntity dst, MemoryEntity src, std::size_t size) const {
        memcpy(dst.ptr, src.ptr, size);
    }   
};

}