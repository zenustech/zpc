#include "LLVM.hpp"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/InitializePasses.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetSelect.h>

#include <iostream>

namespace zs {

  LLVM::LLVM() : _jit{nullptr} {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    auto jit_expected = llvm::orc::LLJITBuilder().create();

    if (!jit_expected) {
      std::cerr << "Zpc-JIT error: failed to create JIT instance: "
                << llvm::toString(jit_expected.takeError()) << std::endl;
    }

    _jit = (*jit_expected).release();
  }

  LLVM::~LLVM() {
    if (_jit) delete _jit;
    _jit = nullptr;
  }

  llvm::orc::LLJIT *LLVM::jit() noexcept { return instance()._jit; }

}  // namespace zs
