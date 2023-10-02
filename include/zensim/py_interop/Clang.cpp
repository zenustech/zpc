#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/InitializePasses.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetSelect.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "zensim/llvm/LLVM.hpp"
#define Zensim_EXPORT
#include "zensim/ZensimExport.hpp"
#include "zensim/ZpcReflection.hpp"

namespace zs {

  static std::unique_ptr<llvm::Module> compile_cpp_to_llvm(const std::string &input_file,
                                                           const char *cpp_src,
                                                           const char *include_dir,
                                                           llvm::LLVMContext &context) {
    std::vector<const char *> args;
    args.push_back("-x");
    args.push_back("c++");
    args.push_back(input_file.c_str());

#if defined(ZS_ENABLE_OPENMP) && ZS_ENABLE_OPENMP
    args.push_back("-fopenmp");
#endif
    args.push_back("-O3");
    args.push_back("-I");
    args.push_back(include_dir);
    args.push_back("-std=c++17");
#if defined(ZS_ENABLE_OPENMP) && ZS_ENABLE_OPENMP
    args.push_back("-DZS_ENABLE_OPENMP=1");
    args.push_back("-DPYZPC_EXEC_TAG=::zs::exec_omp");
#else
    args.push_back("-DPYZPC_EXEC_TAG=::zs::exec_seq");
#endif
    args.push_back("-DZPC_JIT_MODE");

    clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnostic_options
        = new clang::DiagnosticOptions();
    std::unique_ptr<clang::TextDiagnosticPrinter> text_diagnostic_printer
        = std::make_unique<clang::TextDiagnosticPrinter>(llvm::errs(), &*diagnostic_options);
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagnostic_ids;
    std::unique_ptr<clang::DiagnosticsEngine> diagnostic_engine
        = std::make_unique<clang::DiagnosticsEngine>(diagnostic_ids, &*diagnostic_options,
                                                     text_diagnostic_printer.release());

    clang::CompilerInstance compiler_instance;

    auto &compiler_invocation = compiler_instance.getInvocation();
    clang::CompilerInvocation::CreateFromArgs(compiler_invocation, args,
                                              *diagnostic_engine.release());

    // Map code to a MemoryBuffer
    std::unique_ptr<llvm::MemoryBuffer> buffer = llvm::MemoryBuffer::getMemBufferCopy(cpp_src);
    compiler_invocation.getPreprocessorOpts().addRemappedFile(input_file.c_str(), buffer.get());

    // msvc
    compiler_instance.getLangOpts().MicrosoftExt = 1;
    compiler_instance.getLangOpts().DeclSpecKeyword = 1;

    compiler_instance.createDiagnostics(text_diagnostic_printer.get(), false);

    clang::EmitLLVMOnlyAction emit_llvm_only_action(&context);
    bool success = compiler_instance.ExecuteAction(emit_llvm_only_action);
    buffer.release();

    return success ? std::move(emit_llvm_only_action.takeModule()) : nullptr;
  }

}  // namespace zs

extern "C" {

ZENSIM_EXPORT int cpp_compile_program(const char *cpp_src, const char *include_dir,
                                      const char *output_file) {
#if defined(_WIN32)
  const char *obj_ext = ".obj";
#else
  const char *obj_ext = ".o";
#endif

  std::string input_file
      = std::string(output_file).substr(0, std::strlen(output_file) - std::strlen(obj_ext));

  zs::LLVM::maintain(zs::singleton_op_e::init);

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module
      = zs::compile_cpp_to_llvm(input_file, cpp_src, include_dir, context);

  if (!module) {
    return -1;
  }

  std::string target_triple = llvm::sys::getDefaultTargetTriple();
  std::string Error;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(target_triple, Error);

  const char *CPU = "generic";
  const char *features = "";
  llvm::TargetOptions target_options;
  llvm::Reloc::Model relocation_model = llvm::Reloc::PIC_;  // DLLs need Position Independent Code
  llvm::CodeModel::Model code_model
      = llvm::CodeModel::Large;  // Don't make assumptions about displacement sizes
  llvm::TargetMachine *target_machine = target->createTargetMachine(
      target_triple, CPU, features, target_options, relocation_model, code_model);

  module->setDataLayout(target_machine->createDataLayout());

  std::error_code error_code;
  llvm::raw_fd_ostream output(output_file, error_code, llvm::sys::fs::OF_None);

  llvm::legacy::PassManager pass_manager;
  llvm::CodeGenFileType file_type = llvm::CGFT_ObjectFile;
  target_machine->addPassesToEmitFile(pass_manager, output, nullptr, file_type);

  pass_manager.run(*module);
  output.flush();

  delete target_machine;

  return 0;
}

// Global JIT instance
// static llvm::orc::LLJIT *jit = nullptr;

// Load an object file into an in-memory DLL named `module_name`
ZENSIM_EXPORT int load_obj(const char *dll_file, const char *object_file, const char *module_name) {
  auto jit = zs::LLVM::jit();

  auto dll = jit->createJITDylib(module_name);

  if (!dll) {
    std::cerr << "Zpc-JIT error: failed to create JITDylib: " << llvm::toString(dll.takeError())
              << std::endl;
    return -1;
  }

#if defined(_WIN32)
  char global_prefix = '\0';
#elif defined(__APPLE__)
  char global_prefix = '_';
#else
  char global_prefix = '\0';
#endif

  /// omp (/gomp)
#if defined(ZS_ENABLE_OPENMP) && ZS_ENABLE_OPENMP
  {
#  if defined(_WIN32)
    auto search = llvm::orc::DynamicLibrarySearchGenerator::Load("libomp.dll", global_prefix);
#  else
    auto search = llvm::orc::DynamicLibrarySearchGenerator::Load("libomp.so", global_prefix);
#  endif
    if (!search) {
      std::cerr << "Zpc-JIT error: failed to create generator: " << toString(search.takeError())
                << std::endl;
      return -1;
    }
    dll->addGenerator(std::move(*search));
    std::cout << "done load omp runtime" << std::endl;
  }
#endif

  /// explicitly load c-runtime on windows
#if defined(_WIN32)
  {
    auto search = llvm::orc::DynamicLibrarySearchGenerator::Load("ucrtbase.dll", global_prefix);
    if (!search) {
      std::cerr << "Failed to create generator: " << llvm::toString(search.takeError())
                << std::endl;
      return -1;
    }
    dll->addGenerator(std::move(*search));
    std::cout << "done load windows c runtime";
  }
#endif

  /// dll_file: py zpc dynamic library (.dll/.so/.dylib)
  {
    auto search = llvm::orc::DynamicLibrarySearchGenerator::Load(dll_file, global_prefix);

    if (!search) {
      std::cerr << "Zpc-JIT error: failed to create generator: "
                << llvm::toString(search.takeError()) << std::endl;
      return -1;
    }

    dll->addGenerator(std::move(*search));
  }

  // Load the object file into a memory buffer
  auto buffer = llvm::MemoryBuffer::getFile(object_file);
  if (!buffer) {
    std::cerr << "Zpc-JIT error: failed to load object file: " << buffer.getError().message()
              << std::endl;
    return -1;
  }

  auto err = jit->addObjectFile(*dll, std::move(*buffer));
  if (err) {
    std::cerr << "Zpc-JIT error: failed to add object file: " << llvm::toString(std::move(err))
              << std::endl;
    return -1;
  }

  return 0;
}

ZENSIM_EXPORT int unload_obj(const char *module_name) {
  if (!zs::LLVM::initialized()) return 0;

  auto jit = zs::LLVM::jit();
  auto *dll = jit->getJITDylibByName(module_name);
  llvm::Error error = jit->getExecutionSession().removeJITDylib(*dll);

  if (error) {
    std::cerr << "Zpc-JIT error: failed to unload: " << llvm::toString(std::move(error))
              << std::endl;
    return -1;
  }

  return 0;
}

ZENSIM_EXPORT zs::u64 lookup(const char *dll_name, const char *function_name) {
  if (!zs::LLVM::initialized()) throw std::runtime_error("llvm orc jit engine not yet initialied!");
  auto jit = zs::LLVM::jit();
  auto *dll = jit->getJITDylibByName(dll_name);

  auto func = jit->lookup(*dll, function_name);

  if (!func) {
    std::cerr << "Zpc-JIT error: failed to lookup symbol: " << llvm::toString(func.takeError())
              << std::endl;
    return (zs::u64)-1;
  }

  return func->getValue();
}
}