#include "Wrangler.hpp"

#include <filesystem>

#include "zensim/tpls/jitify/jitify2.hpp"
#include "zensim/types/Tuple.h"

namespace fs = std::filesystem;

namespace zs::cudri {

  /// ref: pyb zeno POC
  std::vector<std::string> load_all_ptx_files_at(const std::string &dirpath) {
    std::vector<std::string> res;

    for (auto const &entry : fs::directory_iterator(dirpath)) {
      auto path = entry.path();
      if (fs::path(path).extension() == ".ptx") {
        printf("reading ptx file: %s\n", path.c_str());
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

  void precompile_wranglers(std::string_view progname, std::string_view source) {
    using jitify2::Kernel;
    using jitify2::ProgramCache;
    using jitify2::reflection::Template;
    using jitify2::reflection::Type;
    ;
  }

  void test_jitify(std::string_view progname, std::string_view source) {
    std::string program_name = "my_program";
    std::string program_source = R"(
	    #include <tuple>
  template <typename T, typename T0, typename T1>
  __global__ void my_kernel(T* data, std::tuple<T0, T1> tuple) { 
	  *data = T{7}; printf("%f, %d, %f\n", *data, std::get<0>(tuple), std::get<1>(tuple));
  }
  )";
    dim3 grid(1), block(1);
    float *data;
    cudaMalloc((void **)&data, sizeof(float));
    // jitify2::LoadedProgram program
    jitify2::PreprocessedProgram preprog
        = jitify2::Program(program_name, program_source)
              ->preprocess(
                  {"-std=c++17"});  // Preprocess source code and load all included headers.
    jitify2::ConfiguredKernel configured_kernel
        = preprog
              ->get_kernel(
                  jitify2::reflection::Template("my_kernel").instantiate<float, int, float>())
              ->configure(1, 1);
    configured_kernel->launch(data, std::make_tuple(2, 3.f));

//->configure_1d_max_occupancy()
#if 0
    jitify2::PreprocessedProgramData preprog_data = *preprog;
    // Or we can directly call a method on the data object.
    jitify2::CompiledProgram compiled = preprog->compile("my_kernel");
    compiled->link()
        ->load()
        ->get_kernel("my_kernel<float, int, float>")  // Compile, link, and load the program,
                                                      // and obtain the loaded kernel.
        ->configure(grid, block)                      // Configure the kernel launch.
        ->launch(data, std::make_tuple(2, 3.f));      // Launch the kernel.
    ;
#endif
  }

}  // namespace zs::cudri