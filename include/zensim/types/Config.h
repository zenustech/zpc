#pragma once
#include <zensim/execution/Concurrency.h>

#include "Value.h"
// #include <zensim/execution/ExecutionNode.h>
// #include <zensim/execution/ExecutionPolicy.hpp>
#include <zensim/Singleton.h>

namespace zs {

  struct ExecutionContext : Singleton<ExecutionContext> {
    ExecutionContext() = default;
    ~ExecutionContext() = default;

    struct MethodSetup {
      MethodSetup(const Properties &props) : _props{props} {}
      ConstRefPtr<Property> operator[](std::string tag) const { return _props.find(tag); }
      template <typename T> const T &get(std::string tag) const {
        if (auto it = _props.find(tag); it) return getValue<T>(it->value());
        throw std::runtime_error(
            fmt::format("value extraction error! cannnot find property named [{}]", tag));
      }
      const std::map<std::string, Property> &refProperties() const noexcept {
        return _props.refProperties();
      }

    protected:
      const Properties &_props;
      // execution policy
    };

    static MethodSetup &setup(const std::string &name, const Properties &props) {
      auto &inst = instance();
      if (auto it = inst._methodSetups.find(name); it) {
        inst._active = it;
        return *it;
      }
      /// otherwise instantiate it
      auto it = inst._methodSetups.emplace(name, MethodSetup{props});
      inst._active = &it.first->second;
      return it.first->second;
    }
    static ConstRefPtr<MethodSetup> current() { return instance()._active; }

  protected:
    ConstRefPtr<MethodSetup> _active{nullptr};
    concurrent_map<std::string, MethodSetup> _methodSetups{};
  };

}  // namespace zs