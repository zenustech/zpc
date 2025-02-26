#define Zensim_EXPORT
#include "zensim/ZensimExport.hpp"
#include "zensim/cuda/Cuda.h"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"

//
#include <cstdio>
#include <string>
#include <vector>
//
#include <cuda.h>
// #include <nvPTXCompiler.h>
#include <nvrtc.h>

extern "C" {

///
///
/// compile
///
///
static bool check_nvrtc(nvrtcResult result,
                        const zs::source_location &loc = zs::source_location::current()) {
  if (result == NVRTC_SUCCESS) return true;

  zs::checkCuApiError((zs::u32)result, loc, nvrtcGetErrorString(result));
  return false;
}

ZENSIM_EXPORT size_t cuda_compile_program(const char *cuda_src, int arch, const char *include_dir,
                                          bool debug, bool verbose, bool verify_fp, bool fast_math,
                                          const char *output_path) {
  // determine whether to output PTX or CUBIN depending on the file extension
  const char *output_ext = strrchr(output_path, '.');
  bool use_ptx = output_ext && strcmp(output_ext + 1, "ptx") == 0;

  // check include dir path len (path + option)
  //
  constexpr int max_path = 4096 + 16;
  if (auto len = strlen(include_dir); len > max_path) {
    std::cerr << fmt::format("Zpc-JIT error: include path too long ({})\n", len);
    return (size_t)-1;
  }

  std::string include_opt = "--include-path=";
  include_opt += include_dir;

  std::string arch_opt;
  if (use_ptx)
    arch_opt = fmt::format("--gpu-architecture=compute_{}", arch);
  else
    arch_opt = fmt::format("--gpu-architecture=sm_{}", arch);

  std::vector<const char *> opts;
  opts.push_back(arch_opt.data());
  opts.push_back(include_opt.data());
  opts.push_back("--device-as-default-execution-space");
  opts.push_back("--std=c++17");
  opts.push_back("--define-macro=ZS_ENABLE_CUDA=1");
  opts.push_back("-DPYZPC_EXEC_TAG=zs::zs::cuda_c");
  opts.push_back("-DZPC_JIT_MODE");

  if (debug) {
    opts.push_back("--define-macro=_DEBUG");
    opts.push_back("--generate-line-info");
  } else
    opts.push_back("--define-macro=NDEBUG");

  if (fast_math) opts.push_back("--use_fast_math");

  nvrtcProgram prog;
  nvrtcResult res;

  res = nvrtcCreateProgram(&prog,     // prog
                           cuda_src,  // buffer
                           NULL,      // name
                           0,         // numHeaders
                           NULL,      // headers
                           NULL);     // includeNames

  if (!check_nvrtc(res)) return (size_t)res;

  // nvrtcAddNameExpression(prog, name.c_str());
  res = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());

  // check log
  if (!check_nvrtc(res) || verbose) {
    size_t log_size;
    if (check_nvrtc(nvrtcGetProgramLogSize(prog, &log_size))) {
      std::vector<char> log(log_size);
      if (check_nvrtc(nvrtcGetProgramLog(prog, log.data()))) {
        if (res != NVRTC_SUCCESS)
          std::cerr << fmt::format("{}", log.data());
        else
          std::cout << fmt::format("{}", log.data());
      }
    }
  }
  if (res != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&prog);
    return (size_t)res;
  }

  nvrtcResult (*get_output_size)(nvrtcProgram, size_t *);
  nvrtcResult (*get_output_data)(nvrtcProgram, char *);
  const char *output_mode;
  if (use_ptx) {
    get_output_size = nvrtcGetPTXSize;
    get_output_data = nvrtcGetPTX;
    output_mode = "wt";
  } else {
    get_output_size = nvrtcGetCUBINSize;
    get_output_data = nvrtcGetCUBIN;
    output_mode = "wb";
  }

  // save output
  size_t output_size;
  res = get_output_size(prog, &output_size);
  if (check_nvrtc(res)) {
    std::vector<char> output(output_size);
    res = get_output_data(prog, output.data());
    if (check_nvrtc(res)) {
      FILE *file = fopen(output_path, output_mode);
      if (file) {
        if (fwrite(output.data(), 1, output_size, file) != output_size) {
          std::cerr << fmt::format("Zpc-JIT error: failed to write output file \'{}\'\n",
                                   output_path);
          res = (nvrtcResult)-1;
        }
        fclose(file);
      } else {
        std::cerr << fmt::format("Zpc-JIT error: failed to open output file \'{}\'\n", output_path);
        res = (nvrtcResult)-1;
      }
    }
  }
  check_nvrtc(nvrtcDestroyProgram(&prog));
  return res;
}

///
///
/// run
///
///
ZENSIM_EXPORT void *cuda_load_module(void *pol, const char *path) {
  auto context = ((zs::CudaExecutionPolicy *)pol)->getContext();
  zs::Cuda::ContextGuard guard(context);

  CUdevice device;
  zs::checkCuApiError(cuCtxGetDevice(&device), "[Zpc-JIT::cuCtxGetDevice]");

  // use file extension to determine whether to load PTX or CUBIN
  const char *input_ext = strrchr(path, '.');
  bool load_ptx = input_ext && strcmp(input_ext + 1, "ptx") == 0;

  std::vector<char> input;

  FILE *file = fopen(path, "rb");
  if (file) {
    fseek(file, 0, SEEK_END);
    size_t length = ftell(file);
    fseek(file, 0, SEEK_SET);

    input.resize(length);
    if (fread(input.data(), 1, length, file) != length) {
      std::cerr << fmt::format("Zpc-JIT error: failed to read input file \'{}\'\n", path);
      fclose(file);
      return NULL;
    }
    fclose(file);
  } else {
    std::cerr << fmt::format("Zpc-JIT error: failed to open input file \'{}\'\n", path);
    return NULL;
  }

  int driver_cuda_version = 0;
  CUmodule module = NULL;

  if (load_ptx) {
    if (zs::checkCuApiError(cuDriverGetVersion(&driver_cuda_version),
                            "[Zpc-JIT::cuDriverGetVersion]")
        && driver_cuda_version >= CUDA_VERSION) {
      CUjit_option options[2];
      void *option_vals[2];
      std::string error_log(8192, '\0');
      options[0] = CU_JIT_ERROR_LOG_BUFFER;
      option_vals[0] = (void *)error_log.data();
      options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
      option_vals[1] = (void *)error_log.size();

      if (!zs::checkCuApiError(cuModuleLoadDataEx(&module, input.data(), 2, options, option_vals),
                               "[Zpc-JIT::cuModuleLoadDataEx]")) {
        std::cerr << fmt::format("Zpc-JIT error: loading PTX module failed\n");
        // print error log if not empty
        if (!error_log.empty())
          std::cerr << fmt::format("PTX loader error:\n{}\n", error_log.data());
        return NULL;
      }
    } else {
      // manually compile the PTX and load as CUBIN
      void *state;
      cuLinkCreate(0, nullptr, nullptr, (CUlinkState *)&state);

      // auto jitSrc = cudri::compile_cuda_source_to_ptx(jitCode);
      auto name = fmt::format("[{}]", path);
      cuLinkAddData((CUlinkState)state, CU_JIT_INPUT_PTX, (void *)input.data(),
                    (size_t)input.size(), name.data(), 0, NULL, NULL);

      void *cubin;
      size_t cubinSize;
      cuLinkComplete((CUlinkState)state, &cubin, &cubinSize);

      if (!zs::checkCuApiError(cuModuleLoadData((CUmodule *)&module, cubin),
                               "[Zpc-JIT::cuModuleLoadData]")) {
        std::cerr << fmt::format("Zpc-JIT CUDA error: loading module failed\n");
        cuLinkDestroy((CUlinkState)state);
        return NULL;
      }
      cuLinkDestroy((CUlinkState)state);
    }
  } else {
    // load CUBIN
    if (!zs::checkCuApiError(cuModuleLoadDataEx(&module, input.data(), 0, NULL, NULL),
                             "[Zpc-JIT::cuModuleLoadDataEx]")) {
      std::cerr << fmt::format("Zpc-JIT CUDA error: loading CUBIN module failed\n");
      return NULL;
    }
  }

  return module;
}

ZENSIM_EXPORT void cuda_unload_module(void *pol, void *module) {
  auto context = ((zs::CudaExecutionPolicy *)pol)->getContext();
  zs::Cuda::ContextGuard guard(context);

  zs::checkCuApiError(cuModuleUnload((CUmodule)module), "[Zpc-JIT::cuModuleUnload]");
}

ZENSIM_EXPORT void *cuda_get_kernel(void *pol, void *module, const char *name) {
  auto context = ((zs::CudaExecutionPolicy *)pol)->getContext();
  zs::Cuda::ContextGuard guard(context);

  CUfunction kernel = NULL;
  if (!zs::checkCuApiError(cuModuleGetFunction(&kernel, (CUmodule)module, name),
                           "[Zpc-JIT::cuModuleGetFunction]"))
    fmt::print("Zpc-JIT: failed to lookup kernel function {} in module\n", name);

  return kernel;
}

ZENSIM_EXPORT size_t cuda_launch_kernel(void *context, void *kernel, size_t dim,
                                        void **args, void *stream = nullptr) {
  zs::Cuda::ContextGuard guard(context);

  const int block_dim = 256;
  const int grid_dim = (dim + block_dim - 1) / block_dim;

  CUresult res = cuLaunchKernel((CUfunction)kernel, grid_dim, 1, 1, block_dim,
                                1, 1, 0, (CUstream)stream, args, 0);

  zs::checkCuApiError(res, "[Zpc-JIT::cuLaunchKernel]"); 

  return res;
}

}  // extern "C"
