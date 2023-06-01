#include "Filesystem.hpp"

#include "zensim/zpc_tpls/whereami/whereami.h"

namespace zs {

  /// ref:
  /// https://stackoverflow.com/questions/1023306/finding-current-executables-path-without-proc-self-exe/1024937#1024937
  /// ref: https://gist.github.com/Jacob-Tate/7b326a086cf3f9d46e32315841101109

  std::string abs_exe_path() {
    char *s{};
    int length, dirname_length;
    length = wai_getExecutablePath(NULL, 0, &dirname_length);
    s = (char *)malloc(length + 1);
    wai_getExecutablePath(s, length, &dirname_length);
    s[length] = '\0';
    auto ret = std::string(s);
    free(s);
    return ret;
  }

  std::string abs_exe_directory() {
    char *s{};
    int length, dirname_length;
    length = wai_getExecutablePath(NULL, 0, &dirname_length);
    s = (char *)malloc(length + 1);
    wai_getExecutablePath(s, length, &dirname_length);
    s[dirname_length] = '\0';
    auto ret = std::string(s);
    free(s);
    return ret;
  }

  std::string abs_module_path() {
    char *s{};
    int length, dirname_length;
    length = wai_getModulePath(NULL, 0, &dirname_length);
    s = (char *)malloc(length + 1);
    wai_getModulePath(s, length, &dirname_length);
    s[length] = '\0';
    auto ret = std::string(s);
    free(s);
    return ret;
  }

  std::string abs_module_directory() {
    char *s{};
    int length, dirname_length;
    length = wai_getModulePath(NULL, 0, &dirname_length);
    s = (char *)malloc(length + 1);
    wai_getModulePath(s, length, &dirname_length);
    s[dirname_length] = '\0';
    auto ret = std::string(s);
    free(s);
    return ret;
  }

}  // namespace zs