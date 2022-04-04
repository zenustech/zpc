#pragma once
#include <type_traits>

namespace zs {

  /*
   *	@note	Singleton
   */
  template <typename T> struct Singleton {
    static T &instance() {
/// ref: CppCon 2018: Greg Falcon “Initialization, Shutdown, and constexpr”
// avoid destruction issues
#if 1
      static T *p_instance = new T();
      return *p_instance;
#else
      static T _instance{};
      return _instance;
#endif
    }
  };

}  // namespace zs
