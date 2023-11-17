#pragma once
#include <string>
#include "zensim/Platform.hpp"

namespace zs {

  /// ref:
  /// https://stackoverflow.com/questions/1023306/finding-current-executables-path-without-proc-self-exe/1024937#1024937
  /// ref: https://gist.github.com/Jacob-Tate/7b326a086cf3f9d46e32315841101109
  /// ref: https://github.com/gpakosz/whereami

  ZPC_BACKEND_API std::string abs_exe_path();
  ZPC_BACKEND_API std::string abs_exe_directory();
  ZPC_BACKEND_API std::string abs_module_path();
  ZPC_BACKEND_API std::string abs_module_directory();

}  // namespace zs