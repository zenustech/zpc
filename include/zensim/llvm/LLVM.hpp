#pragma once

#include "zensim/Singleton.h"

namespace llvm {
  namespace orc {
    class LLJIT;
  }
}  // namespace llvm

namespace zs {

  struct LLVM : Singleton<LLVM> {
  public:
    LLVM();
    ~LLVM();

    static llvm::orc::LLJIT *jit() noexcept;

  private:
    llvm::orc::LLJIT *_jit = nullptr;
  };

}  // namespace zs
