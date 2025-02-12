#pragma once

#include "zensim/Platform.hpp"

// #if !defined(ZS_COMPILER_SYCL_VER)
#if !ZS_ENABLE_SYCL
#  error "ZS_ENABLE_SYCL* was not enabled, but Sycl.hpp was included anyway."
#endif

#include <sycl/sycl.hpp>

namespace zs {

  struct Sycl {
  private:
    Sycl();

  public:
    ZPC_BACKEND_API static Sycl &instance() {
      static Sycl s_instance{};
      return s_instance;
    }

    struct ZPC_BACKEND_API SyclContext {
    };

    sycl::context _defaultCtx;
  };

}  // namespace zs