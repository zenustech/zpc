#pragma once
#include <string>
#include <string_view>
#include <vector>

namespace zs::cudri {

  /// jitify: preprocess() -> compile() -> link() -> load()
  std::vector<std::string> load_all_ptx_files_at(const std::string &dirpath = ZS_PTX_INCLUDE_DIR);
  std::string compile_cuda_source_to_ptx(std::string_view code, std::string_view name = "unnamed",
                                         std::vector<std::string_view> additionalOptions = {});
  void precompile_wranglers(std::string_view progname, std::string_view source);

  void test_jitify(std::string_view progname = "", std::string_view source = "");

}  // namespace zs::cudri