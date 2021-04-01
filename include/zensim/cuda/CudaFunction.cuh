#pragma once
#include <mutex>
#include <string>

#include "../Logger.hpp"
#include "../Platform.hpp"
#include "zensim/tpls/fmt/core.h"

namespace zs {

  std::string get_cu_error_message(uint32_t err);
  std::string get_cuda_error_message(uint32_t err);

  template <typename... Args> class CudaDriverApi {
  public:
    CudaDriverApi() { function = nullptr; }

    void set(void *func_ptr) { function = (func_type *)func_ptr; }

    uint32_t call(Args... args) {
      assert(function != nullptr);
      assert(driver_lock != nullptr);
      std::lock_guard<std::mutex> _(*driver_lock);
      return (uint32_t)function(args...);
    }

    void set_names(const std::string &name, const std::string &symbol_name) {
      this->name = name;
      this->symbol_name = symbol_name;
    }

    void set_lock(std::mutex *lock) { driver_lock = lock; }

    std::string get_error_message(uint32_t err) {
      return get_cu_error_message(err) + fmt::format(" while calling {} ({})", name, symbol_name);
    }

    uint32_t call_with_warning(Args... args) {
      auto err = call(args...);
      ZS_WARN_IF(err, "{}", get_error_message(err));
      return err;
    }

    // Note: CUDA driver API passes everything as value
    void operator()(Args... args) {
      auto err = call(args...);
      ZS_ERROR_IF(err, get_error_message(err));
    }

  private:
    using func_type = uint32_t(Args...);

    func_type *function{nullptr};
    std::string name, symbol_name;
    std::mutex *driver_lock{nullptr};
  };

  template <typename... Args> class CudaRuntimeApi {
  public:
    CudaRuntimeApi() { function = nullptr; }

    void set(void *func_ptr) { function = (func_type *)func_ptr; }

    uint32_t call(Args... args) {
      assert(function != nullptr);
      assert(driver_lock != nullptr);
      std::lock_guard<std::mutex> _(*driver_lock);
      return (uint32_t)function(args...);
    }

    void set_names(const std::string &name, const std::string &symbol_name) {
      this->name = name;
      this->symbol_name = symbol_name;
    }

    void set_lock(std::mutex *lock) { driver_lock = lock; }

    std::string get_error_message(uint32_t err) {
      return get_cuda_error_message(err) + fmt::format(" while calling {} ({})", name, symbol_name);
    }

    uint32_t call_with_warning(Args... args) {
      auto err = call(args...);
      ZS_WARN_IF(err, "{}", get_error_message(err));
      return err;
    }

    void operator()(Args... args) {
      auto err = call(args...);
      ZS_ERROR_IF(err, get_error_message(err));
    }

  private:
    using func_type = uint32_t(Args...);

    func_type *function{nullptr};
    std::string name, symbol_name;
    std::mutex *driver_lock{nullptr};
  };

}  // namespace zs
