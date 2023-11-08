#include "Wrangler.hpp"

#include <cuda.h>
#include <nvrtc.h>

#include <filesystem>

#include "zensim/cuda/Cuda.h"
#include "zensim/io/Filesystem.hpp"
#include "zensim/types/Tuple.h"
#include "zensim/zpc_tpls/fmt/format.h"
#include "zensim/zpc_tpls/jitify/jitify2.hpp"

namespace fs = std::filesystem;

namespace zs::cudri {

  /// ref: pyb zeno POC
  std::vector<std::string> load_all_ptx_files_at(const std::string &localPath) {
#if RESOURCE_AT_RELATIVE_PATH
    auto dirpath = abs_exe_directory() + "/" + localPath;
#else
    auto dirpath = std::string{AssetDirPath} + "/" + localPath;
#endif
    std::vector<std::string> res;

    if (!std::filesystem::exists(dirpath))
      throw std::runtime_error(
          fmt::format("cannot find directory [{}] for loading ptx files.\n", dirpath));
    for (auto const &entry : fs::directory_iterator(dirpath)) {
      auto path = entry.path();
      if (fs::path(path).extension() == ".ptx") {
        fmt::print("reading ptx file: {}\n", path.string());
        std::ifstream fin(path, std::ios::in | std::ios::binary | std::ios::ate);
        if (!fin.is_open()) {
          std::cerr << "\nerror: unable to open " << path << " for reading!\n";
          abort();
        }

        size_t inputSize = (size_t)fin.tellg();
        char *memBlock = new char[inputSize + 1];

        fin.seekg(0, std::ios::beg);
        fin.read(memBlock, inputSize);
        fin.close();

        memBlock[inputSize] = '\0';
        res.emplace_back(memBlock);
        delete memBlock;
      }
    }
    return res;
  }
  std::string compile_cuda_source_to_ptx(std::string_view code, std::string_view name,
                                         std::vector<std::string_view> additionalOptions) {
    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);

    std::vector<std::string> fixedOpts{
        fmt::format("--include-path={}", ZS_INCLUDE_DIR), "--device-as-default-execution-space",
        fmt::format("--gpu-architecture=sm_{}{}", major, minor), "-dc", "-std=c++17"};
    std::vector<const char *> opts(fixedOpts.size() + additionalOptions.size());
    size_t loc = 0;
    for (auto &&opt : fixedOpts) opts[loc++] = opt.data();
    for (auto &&opt : additionalOptions) opts[loc++] = opt.data();

    const char *userScript = code.data();

    nvrtcProgram prog;
    #if 0
    const char *headers[] = {"type_traits", "initializer_list"};
    nvrtcCreateProgram(&prog, userScript, name.data(), 2, headers, NULL);
    #else
    nvrtcCreateProgram(&prog, userScript, name.data(), 0, nullptr, NULL);
    #endif
    nvrtcResult res = nvrtcCompileProgram(prog, opts.size(), opts.data());

    size_t strSize{0};
    std::string str{};
    nvrtcGetProgramLogSize(prog, &strSize);
    str.resize(strSize + 1);
    nvrtcGetProgramLog(prog, str.data());
    str[strSize] = '\0';
    if (str.size() >= 3) fmt::print("\n compilation log ---\n{}\n end log ---\n", str);

    nvrtcGetPTXSize(prog, &strSize);
    str.resize(strSize + 1);
    nvrtcGetPTX(prog, str.data());
    str[strSize] = '\0';
    return str;
  }

}  // namespace zs::cudri