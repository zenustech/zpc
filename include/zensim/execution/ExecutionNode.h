#pragma once
#include <zensim/tpls/fmt/color.h>
#include <zensim/meta/Functional.h>
#include <zensim/types/Function.h>
#include <zensim/types/Node.h>
#include <zensim/types/Tuple.h>

#include <exception>
#include <system_error>
#include <zensim/types/Event.hpp>

#include "Reflection.h"

namespace zs {

  template <typename Signature> struct ExecNode;
  template <typename R, typename... Args> ExecNode(std::function<R(Args...)>)
      // -> ExecNode<add_optional_t<R>(add_optional_t<Args>...)>;
      ->ExecNode<R(Args...)>;
  template <typename R> ExecNode(std::function<R()>)
      // -> ExecNode<add_optional_t<R>(optional<void>)>;
      ->ExecNode<R(void)>;
  using task_t = ExecNode<void(void)>;

  template <typename... Signature>  ///< self + params (optional)
  struct TaskNode;
  template <typename R, typename... Args, typename... TaskNodes>
  TaskNode(ExecNode<R(Args...)> &node, TaskNodes *...taskNodes)
      -> TaskNode<R(Args...), TaskNodes...>;
  template <typename R, typename... Args> TaskNode(ExecNode<R(Args...)> &node)
      -> TaskNode<R(Args...)>;

  /// generator
  template <typename Derived, typename Signature, typename Indices> struct ExecNodeInterface;
  template <typename Derived, typename R, typename... Args, std::size_t... Is>
  struct ExecNodeInterface<Derived, R(Args...), index_seq<Is...>> {
    constexpr Derived const &self() const noexcept { return static_cast<Derived const &>(*this); }
    add_optional_t<R> operator()(const add_optional_t<Args> &...args) const {
      if ((args.has_value() && ...)) {
        try {
          return add_optional_t<R>{std::invoke(self().f(), (*args)...)};
        } catch (std::exception const &e) {
          fmt::print(fg(fmt::color::red), "Node Execution Error: {}\n", e.what());
          return nullopt;
        }
      }
      return nullopt;
    }
    add_optional_t<R> &operator()(const add_optional_t<Args> &...args) {
      if ((args.has_value() && ...)) {
        try {
          add_optional_t<R>{std::invoke(self().f(), (*args)...)}.swap(_ret);
          return _ret;
        } catch (std::exception const &e) {
          fmt::print(fg(fmt::color::red), "Node Execution Error: {}\n", e.what());
          return _ret;
        }
      }
      return _ret;
    }
    /// is_same_v<type_seq<ChildNodes...>::type<Is>::result_t,
    /// add_optional_t<Args>>>
    template <typename... TaskNodes> auto task(tuple<TaskNodes *...> params) {
      return [params, this]() {
        // (*this) should not be const-decorated
        _ret = std::invoke((*this), params.template get<Is>()->_execNodeBinding._ret...);
      };
    }

    add_optional_t<R> _ret{};
  };

  template <typename R, typename... Args> struct ExecNode<R(Args...)>
      : Inherit<Node, ExecNode<R(Args...)>>,
        ExecNodeInterface<ExecNode<R(Args...)>, R(Args...), std::index_sequence_for<Args...>> {
    using func_t = std::function<R(Args...)>;
    using param_t = tuple<const add_optional_t<Args> &...>;
    using result_t = add_optional_t<R>;
    ExecNode(func_t const &f) : _f{f} {}
    func_t const &f() const { return _f; }

  protected:
    func_t _f;
  };

  /// memory space
  /// execution policy
  /// event, listener
  struct ExecResult {
    ExecResult &message(const std::string &msg) {
      _message = msg;
      return *this;
    }
    const auto &message() { return _message; }

    void display() const { fmt::print("message: {}\n", _message); }

  protected:
    bool _modified{false};
    optional<std::error_code> _error{};
    std::string _message{};
  };
  using TaskExecEvent = Event<ExecResult>;  ///< execution
  ZS_type_name(TaskExecEvent);

  struct TaskGraphNode;
  ZS_type_name(TaskGraphNode);
  struct TaskGraphNode : Inherit<Node, TaskGraphNode> {
    using task_t = function<void()>;
    using param_t = tuple<std::vector<std::string>>;
    using children_t = std::vector<TaskGraphNode *>;

    TaskGraphNode() : _resStr{}, _f{}, _children{}, _listeners{}, _event{} {}
    TaskGraphNode(task_t const &f, std::string resStr = {})
        : _resStr{std::move(resStr)}, _f{f}, _children{}, _listeners{}, _event{} {}

    task_t const &f() const { return _f; }

    /// result string refers to a variable indexed by a string
    void setResultString(std::string &&str) noexcept { _resStr = std::move(str); }
    void setResultString(const std::string &str) { _resStr = str; }
    const std::string &getResultString() const noexcept { return _resStr; }

    void depends(TaskGraphNode &node) {
      _children.push_back(&node);
      _listeners.push_back(
          node.refEvent().createListener([](const ExecResult &res) { res.display(); }));
    }
    void depends(std::size_t id, TaskGraphNode &node) {
      if (id >= _children.size()) return;
      _children[id] = &node;
      dynamic_cast<TaskExecEvent::Listener &>(_listeners[id]).reset();
      _listeners[id] = node.refEvent().createListener([](const ExecResult &res) { res.display(); });
    }

    auto getTask() const -> task_t {
      return [this]() -> void {
        try {
          std::invoke(_f);
          _event.emit(
              ExecResult{}.message(fmt::format("success in computing task on {}\n", _resStr)));
        } catch (std::exception const &e) {
          fmt::print(fg(fmt::color::red), "TaskGraphNode Execution Error: {}\n", e.what());
          _event.emit(
              ExecResult{}.message(fmt::format("error in computing task on {}\n", _resStr)));
          /// SPDLOG::
        }
      };
    }
    const children_t &getChildren() const noexcept { return _children; }
    TaskExecEvent &refEvent() noexcept { return _event; }

  protected:
    std::string _resStr;
    task_t _f;
    children_t _children;
    std::vector<Listener> _listeners;
    TaskExecEvent _event;
  };
  template <typename> struct is_taskgraphnode : std::false_type {};
  template <> struct is_taskgraphnode<TaskGraphNode> : std::true_type {};

  template <typename R, typename... Args, typename... TaskNodes>
  struct TaskNode<R(Args...), TaskNodes...> : Inherit<Node, TaskNode<R(Args...), TaskNodes...>> {
    using result_t = typename ExecNode<R(Args...)>::result_t;
    using Params = type_seq<Args...>;
    using Children = type_seq<TaskNodes...>;
    static constexpr auto nparams = sizeof...(TaskNodes);

    TaskNode(ExecNode<R(Args...)> &node, TaskNodes *...taskNodes)
        : _execNodeBinding{node}, _taskBindings{taskNodes...} {}

    template <placeholders::placeholder_type I>
    void bind(wrapv<I>, typename Children::template type<I> *childTask) {
      static_assert(is_same_v<typename Params::template type<I>,
                              typename Children::template type<I>::result_t>,
                    "binding tasknode return-type mismatch\n");
      _taskBindings.template get<I>() = childTask;
    }
    operator ExecNode<R(Args...)> &() { return _execNodeBinding; }
    ExecNode<R(Args...)> &_execNodeBinding;
    tuple<TaskNodes *...> _taskBindings;
  };
  template <typename> struct is_tasknode : std::false_type {};
  template <typename... Signatures> struct is_tasknode<TaskNode<Signatures...>> : std::true_type {};

  /// with children

}  // namespace zs