#pragma once
#include <type_traits>

namespace zs {

  /*
   *	@note	Singleton
   */
  template <typename T> struct Singleton {
    static T &instance() {
      static T _instance{};
      return _instance;
    }
  };

}  // namespace zs
