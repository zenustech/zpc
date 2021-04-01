#pragma once
#include <mutex>
#include <type_traits>

#include "Event.hpp"
#include "Value.h"
#include "zensim/container/Vector.hpp"
#include "zensim/execution/Concurrency.h"
#include "zensim/types/Object.h"
#include "zensim/types/Tuple.h"

namespace zs {

  struct Node;
  ZS_type_name(Node);
  using NodeOwner = std::unique_ptr<Node>;
  using NodeObserver = RefPtr<Node>;
  using ConstNodeObserver = ConstRefPtr<Node>;

  struct Node : Object {
    Node() noexcept = default;
    Node(NodeObserver parent) noexcept : _parent{parent} {}
    Node(Node &&) noexcept = default;
    Node &operator=(Node &&) noexcept = default;
    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;
    Node(ObjOwner &&o, NodeObserver parent = nullptr) noexcept
        : Object{std::move(*o)}, _parent{parent} {}

    // nodeobserver is a pointer, thus const here is meaningful
    constexpr NodeObserver getParent() noexcept { return _parent; }
    constexpr ConstNodeObserver getParent() const noexcept { return _parent; }
    void setParent(NodeObserver parent) noexcept { _parent = parent; }

  protected:
    NodeObserver _parent{nullptr};
  };

  struct GroupNode : Node {
    using Children = std::vector<NodeOwner>;
    using ChildIndex = std::size_t;

    GroupNode() noexcept = default;
    GroupNode(NodeObserver parent, ChildIndex childCnt = 0) : Node{parent}, _children(childCnt) {}
    GroupNode(GroupNode &&) noexcept = default;
    GroupNode &operator=(GroupNode &&) noexcept = default;
    GroupNode(const GroupNode &) = delete;
    GroupNode &operator=(const GroupNode &) = delete;

    constexpr Children &getChildren() noexcept { return _children; }
    constexpr const Children &getChildren() const noexcept { return _children; }

    ~GroupNode() override {
      for (auto &obj : _children) obj->unregister();
    }

    template <typename T> NodeObserver addChildNode(std::unique_ptr<T> &&child) {
      if (auto namePtr = Object::tryGet(this); namePtr) {
        const auto id = _children.size();
        const auto childName = *namePtr + ".child" + std::to_string(id);
        _children.emplace_back(new Node(std::move(child)));
        _children.back()->trackBy(childName);
        return _children.back().get();
      }
      ZS_WARN("father node does not have a name yet!");
      return nullptr;
    }
    NodeObserver addChildNode(std::string className) {
      if (std::string *namePtr = Object::tryGet(this); namePtr) {
        const auto id = _children.size();
        const auto childName = *namePtr + ".child" + std::to_string(id);
        auto child = Instancer::create(className, childName);
        _children.emplace_back(new Node(std::move(child)));
        return _children.back().get();
      }
      ZS_WARN("father node does not have a name yet!");
      return nullptr;
    }

  protected:
    Children _children{};
  };

  struct MethodNode : GroupNode {
    using Children = GroupNode::Children;
    using ChildIndex = GroupNode::ChildIndex;
    // using Attributes = std::vector<ObjOwner>;
    using Bindings = std::vector<ObjObserver>;
    using Params = std::vector<std::string>;

    MethodNode() noexcept = default;
    MethodNode(MethodNode &&) noexcept = default;
    MethodNode &operator=(MethodNode &&) noexcept = default;
    MethodNode(const MethodNode &) = delete;
    MethodNode &operator=(const MethodNode &) = delete;
    /// currently only support free functions & functors/ lambdas
    template <typename F>
    MethodNode(std::string name, F &&f, NodeObserver parent = nullptr, ChildIndex childCount = 0)
        : GroupNode{parent, childCount}, _params(zs::function_traits<F>::arity, nullptr) {
      emplace(std::forward<F>(f));
      trackBy(name);
      _result->trackBy(name + ".result");
    }
    ~MethodNode() override {
      if (_result) _result->unregister();
    }

    template <typename F> void emplace(F &&f) {
      using Callable = std::decay_t<F>;
      using Traits = zs::function_traits<F>;
      using Ret = typename Traits::return_t;
      using Args = typename Traits::arguments_t;
      static_assert(
          std::is_default_constructible<Ret>::value && std::is_move_assignable<Ret>::value,
          "result should be default-constructible and move-assignable!");

      _method = std::make_unique<Method<Callable, Ret, Args>>(*this, std::forward<F>(f));
      _prefixNodes.resize(Traits::arity, nullptr);
      if (_result) _result->unregister();
      _result = std::make_unique<Ret>();
    }

    void addProperties(std::initializer_list<std::pair<std::string, Property &&>> properties);
    void addProperty(std::pair<std::string, Property &&> taggedProp) {
      _attributes.add(std::move(taggedProp.first), std::move(taggedProp.second));
    }
    void addProperty(std::string attribName, Property &&prop) {
      _attributes.add(std::move(attribName), std::move(prop));
    }
    template <typename T>
    void addProperty(std::string attribName, T &&v, prop_access_e tag = prop_access_e::rw) {
      _attributes.add(std::move(attribName), FWD(v), tag);
    }
    void setProperty(std::string attribName, const Property &prop) {
      _attributes.set(std::move(attribName), prop);
    }
    template <typename T>
    void setProperty(std::string attribName, const T &v, prop_access_e tag = prop_access_e::rw) {
      _attributes.set(std::move(attribName), v, tag);
    }
    bool ready() const noexcept {
      for (auto &param : _params)
        if (param == nullptr) return false;
      return true;
    }
    [[nodiscard]] bool matchBindingType(std::size_t id, const ObjObserver obj) const noexcept;
    [[nodiscard]] bool bind(std::size_t id, std::string varName) noexcept;
    [[nodiscard]] bool bind(std::size_t id, MethodNode &node) noexcept;
    void unbind(std::size_t id) noexcept {
      _params[id] = nullptr;
      _prefixNodes[id] = nullptr;
    }

    void operator()(const Params &params) { (*_method)(params); }
    void operator()() { (*_method)(); }

    auto &refPrefixNodes() noexcept { return _prefixNodes; }
    const auto &refPrefixNodes() const noexcept { return _prefixNodes; }
    ObjObserver getResult() noexcept { return _result.get(); }
    ConstObjObserver getResult() const noexcept { return _result.get(); }
    ObjOwner &refResult() noexcept { return _result; }
    const ObjOwner &refResult() const noexcept { return _result; }
    std::string getResultName() const { return Object::track(_result.get()); }
    std::string getNodeName() const { return Object::track((ObjObserver)this); }
    void setupExecutionContext() const;

  private:
    struct MethodInterface {
      virtual ~MethodInterface() = default;
      virtual void operator()(const Params &params) = 0;
      virtual void operator()() = 0;
      virtual bool matchBindingType(const std::size_t, const ObjObserver) const noexcept = 0;
    };

    template <typename Callable, typename R, typename Args> struct Method;
    template <typename Callable, typename R, typename... Args>
    struct Method<Callable, R, std::tuple<Args...>> : MethodInterface {
      static_assert((is_object<std::decay_t<Args>>::value && ...) && is_object<R>::value,
                    "method arguments should all be inherited from Object!");
      static_assert((!std::is_rvalue_reference_v<Args> && ...),
                    "method arguments should not be rvalue reference!");

      using arguments_t = std::tuple<Args...>;

      template <typename F> Method(MethodNode const &method, F &&f)
          : _method{method}, _f{std::forward<F>(f)} {}

    private:
      template <size_t... I>
      void execute(const std::tuple<std::decay_t<Args> &...> &t, std::index_sequence<I...>) {
        _method.setupExecutionContext();
        *(dynamic_cast<R *>(_method._result.get())) = std::invoke(_f, std::get<I>(t)...);
      }
      template <std::size_t... Is>
      optional<std::tuple<std::decay_t<Args> &...>> translate(const Params &params,
                                                              index_seq<Is...>) const {
        if (params.size() != sizeof...(Args)) return {};
        try {
          std::tuple<std::decay_t<Args> *...> paramPtrs{
              dynamic_cast<std::decay_t<Args> *>(Object::track(params[Is]))...};
          if (((std::get<Is>(paramPtrs) == nullptr) || ...)) {
            auto msg = (std::string{"Method parameter cast failure: "} + ...
                        + fmt::format("{} ", (void *)std::get<Is>(paramPtrs)));
            throw std::runtime_error(msg);
          }
          std::tuple<std::decay_t<Args> &...> paramTuple{*std::get<Is>(paramPtrs)...};
          return optional<std::tuple<std::decay_t<Args> &...>>{std::move(paramTuple)};
        } catch (std::exception const &e) {
          ZS_WARN(e.what());
          return {};
        }
      }
      template <std::size_t... Is>
      optional<std::tuple<std::decay_t<Args> &...>> translate(index_seq<Is...>) const {
        for (const auto &objPtr : _method._params)
          if (!objPtr) return {};  ///< unbinded parameter
        std::tuple<std::decay_t<Args> &...> paramTuple{
            (*static_cast<std::decay_t<Args> *>(_method._params[Is]))...};
        return optional<std::tuple<std::decay_t<Args> &...>>{std::move(paramTuple)};
      }
      template <std::size_t I = 0>
      bool matchBindingTypeImpl(const std::size_t index, const ObjObserver param) const noexcept {
        if constexpr (I < sizeof...(Args)) {
          using Expected = std::decay_t<std::tuple_element_t<I, arguments_t>>;
          if (I == index) return dynamic_cast<const Expected *>(param) != nullptr;
          return matchBindingTypeImpl<I + 1>(index, param);
        }
        return false;
      }

    public:
      bool matchBindingType(const std::size_t index,
                            const ObjObserver param) const noexcept override {
        return matchBindingTypeImpl<0>(index, param);
      }
      void operator()(const Params &params) override {
        auto paramTuple = translate(params, std::index_sequence_for<Args...>{});
        if (paramTuple.has_value()) {
          try {
            execute(*paramTuple, std::index_sequence_for<Args...>{});
            return;
          } catch (std::exception const &e) {
            ZS_WARN(e.what());
            return;
          }
        }
        ZS_WARN("Failed method execution due to input parameter translation");
        return;
      }
      void operator()() override {
        auto paramTuple = translate(std::index_sequence_for<Args...>{});
        if (paramTuple.has_value()) {
          try {
            execute(*paramTuple, std::index_sequence_for<Args...>{});
            return;
          } catch (std::exception const &e) {
            ZS_WARN(e.what());
            return;
          }
        }
        ZS_WARN("Failed method execution due to input parameter translation");
        return;
      }

    private:
      MethodNode const &_method;
      Callable _f;
    };

    struct ExecutionResult {
      ExecutionResult &message(const std::string &msg) {
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
    using MethodExecutionEvent = Event<ExecutionResult>;  ///< execution

  protected:
    /// node graph
    std::vector<RefPtr<MethodNode>> _prefixNodes;
    std::vector<Listener> _listeners;
    MethodExecutionEvent _event;
    /// method
    std::unique_ptr<MethodInterface> _method;
    // auxiliary settings should be either:
    // 1. stored within method callable
    // 2. passed in through bindings as a data aggregator
    Properties _attributes{};  // or use vector?
    /// signature
    Bindings _params{};  ///< method input signature
    ObjOwner _result{};
  };

  template <typename> struct is_method_node : std::false_type {};
  template <> struct is_method_node<MethodNode> : std::true_type {};

  namespace script {
    /// indexing
    ObjObserver get_object(const std::string &objectName);
    RefPtr<MethodNode> get_method_node(const std::string &methodName);

    /// creation
    void addObject(std::string className, std::string objectName);
    template <typename F> void addMethod(std::string methodName, F &&f) {
      Holder<MethodNode> nodePtr = std::make_unique<MethodNode>(methodName, FWD(f));
      Object::globalObjects().emplace(methodName, std::move(nodePtr));
    }
    /// value
    void setValue(std::string varName, const PresetValue &value);
    /// binding
    void bindNode(std::string inObject, int id, std::string outObject);
    void bindObject(std::string inObject, int id, std::string var);
    /// property setup
    void setProperty(std::string methodName, std::string propertyName, const PresetValue &value);
    template <typename V, enable_if_t<is_object<V>::value> = 0>
    void setProperty(std::string methodName, std::string propertyName, V &&value) {
      auto &node = *static_cast<MethodNode *>(Object::globalObjects().get(methodName).get());
      node.addProperty(propertyName, FWD(value));
    }
    /// execution
    void display(std::string methodName);
  }  // namespace script

  struct MethodInstancer {
  private:
    struct Concept {
      virtual ~Concept() = default;
      virtual Holder<MethodNode> new_method(const std::string &) const = 0;
    };
    template <typename Callable> struct Builder : Concept {
      static inline std::once_flag registerFlag{};
      template <typename F> Builder(F &&f) : _f{FWD(f)} {}
      //
      Holder<MethodNode> new_method(const std::string &methodName) const override {
        return std::make_unique<MethodNode>(methodName, _f);
      }

    protected:
      Callable _f;
    };
    using BuilderPtr = std::unique_ptr<Concept>;
    inline static concurrent_map<std::string, BuilderPtr> _methodMap;

  public:
    template <typename F> MethodInstancer(std::string name, F &&f) {
      using InstanceBuilder = Builder<std::decay_t<F>>;
      std::call_once(InstanceBuilder::registerFlag, [className = std::move(name), func = FWD(f)]() {
        // only register once (when empty)
        if (_methodMap.find(className) == nullptr) {
          ZS_TRACE("registering method class [{}]\n", className);
          _methodMap.emplace(std::move(className),
                             std::move(std::make_unique<InstanceBuilder>(func)));
        } else
          throw std::runtime_error(
              fmt::format("Another method class has already been registered "
                          "with the same name [{}]",
                          className));
      });
    }
    static Holder<MethodNode> create(std::string className, std::string methodName) {
      if (const auto creator = _methodMap.find(className); creator) {
        ZS_TRACE("creating method instance [{}]: [{}]\n", className, methodName);
        return (*creator)->new_method(methodName);
      }
      throw std::runtime_error(fmt::format(
          "Method classname[{}] used for creating a instance does not exist!", className));
    }
  };

  namespace script {
    /// register
    template <typename F> void registerMethod(std::string methodClassName, F &&f) {
      (void)MethodInstancer(methodClassName, FWD(f));
    }
    /// creation
    void addMethodNode(std::string methodName, std::string methodClassName);
    /// execute
    void apply(std::string methodName,
               const std::vector<std::pair<std::string, PresetValue>> &properties = {});
  }  // namespace script

}  // namespace zs