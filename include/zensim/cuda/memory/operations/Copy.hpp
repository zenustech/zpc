#pragma once
#include "zensim/cuda/Cuda.h"
#include "zensim/memory/operations/Copy.hpp"

namespace zs {

template <>
struct mem_copy<execspace_e::cuda> {
    void operator()(MemoryEntity dst, MemoryEntity src, std::size_t size) const {
        cudaMemcpy(dst.ptr, src.ptr, size, cudaMemcpyDefault);
    }   
};

}