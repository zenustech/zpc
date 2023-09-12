#pragma once
#include <stdexcept>

namespace zs {

  /*
   *	@note	Singleton
   */
  enum singleton_op_e { init = 0, reinit, deinit };
  template <typename T> struct Singleton {
    /// @brief allow early deinitialization, recreation, initialization with params, etc.
    static T *maintain(singleton_op_e op) {
      /// ref: CppCon 2018: Greg Falcon “Initialization, Shutdown, and constexpr”
      // avoid destruction issues
      static T *p_instance = new T();
      // create if instance is absent
      if (op == singleton_op_e::init) {
        if (p_instance == nullptr) p_instance = new T();
      } else if (op == singleton_op_e::reinit) {
        if (p_instance) delete p_instance;
        p_instance = new T();
      } else if (op == singleton_op_e::deinit) {
        if (p_instance) delete p_instance;
        p_instance = nullptr;
      } else {
        throw std::runtime_error("Please use a valid op index for maintenance!");
      }
      return p_instance;
    }

    static T &instance() { return *maintain(singleton_op_e::init); }
  };

}  // namespace zs
