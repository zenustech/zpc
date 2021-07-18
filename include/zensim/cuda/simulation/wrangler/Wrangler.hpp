#pragma once
#include <string>
#include <string_view>
#include <vector>

namespace zs::cudri {

  /// jitify: preprocess() -> compile() -> link() -> load()
  std::vector<std::string> load_all_ptx_files_at(const std::string &dirpath = ZS_PTX_INCLUDE_DIR);
  void precompile_wranglers(std::string_view progname, std::string_view source);

  void test_jitify(std::string_view progname = "", std::string_view source = "");

}  // namespace zs::cudri