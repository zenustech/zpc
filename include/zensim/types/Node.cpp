#include "Node.h"

#include "Config.h"

namespace zs {

  void MethodNode::addProperties(
      std::initializer_list<std::pair<std::string, Property &&>> properties) {
    for (auto &taggedProperty : properties)
      _attributes.add(std::move(taggedProperty.first), std::move(taggedProperty.second));
  }
  bool MethodNode::matchBindingType(std::size_t id, const ObjObserver obj) const noexcept {
    return _method->matchBindingType(id, obj);
  }
  bool MethodNode::bind(std::size_t id, std::string varName) noexcept {
    if (id >= _params.size()) return false;
    if (auto it = Object::tryGet(varName); it)
      if (matchBindingType(id, *it)) {
        ZS_TRACE("MethodNode[{}]: binding {}-th input parameter with object [{}]", getNodeName(),
                 id, varName);
        _params[id] = *it;
        return true;
      }
    return false;
  }
  bool MethodNode::bind(std::size_t id, MethodNode &node) noexcept {
    if (id >= _params.size()) return false;
    if (auto obj = node.getResult(); matchBindingType(id, obj)) {
      ZS_TRACE(
          "MethodNode[{}]: binding {}-th input parameter with node "
          "[{}]\'s result [{}]",
          getNodeName(), id, node.getNodeName(), node.getResultName());
      _params[id] = obj;
      _prefixNodes[id] = &node;
      // setParent(static_cast<NodeObserver>(&node));// parent means a higher
      // hierarchy
      return true;
    }
    return false;
  }
  void MethodNode::setupExecutionContext() const {
    ExecutionContext::setup(objectName(), _attributes);
  }

  namespace script {
    ObjObserver get_object(const std::string &objectName) {
      return static_cast<ObjObserver>(Object::globalObjects().get(objectName).get());
    }
    RefPtr<MethodNode> get_method_node(const std::string &methodName) {
      return static_cast<MethodNode *>(Object::globalObjects().get(methodName).get());
    }
    void addObject(std::string className, std::string objectName) {
      Object::globalObjects().emplace(objectName, Instancer::create(className, objectName));
    }
    void addMethodNode(std::string methodName, std::string methodClassName) {
      Holder<MethodNode> nodePtr = MethodInstancer::create(methodClassName, methodName);
      Object::globalObjects().emplace(methodName, std::move(nodePtr));
    }
    void setValue(std::string varName, const PresetValue &value) {
      match([&varName](const auto &v) {
        using V = std::decay_t<decltype(v)>;
        auto &val = *static_cast<Value<V> *>(get_object(varName));
        val.set(v);
      })(value);
    }
    void bindNode(std::string iName, int id, std::string oName) {
      auto &outNode = *get_method_node(oName);
      auto &inNode = *get_method_node(iName);
      inNode.bind(id, outNode);
    }
    void bindObject(std::string iName, int id, std::string var) {
      auto &inNode = *get_method_node(iName);
      inNode.bind(id, var);
    }
    void setProperty(std::string methodName, std::string propertyName, const PresetValue &value) {
      match([&methodName, &propertyName](const auto &v) {
        auto &node = *get_method_node(methodName);
        node.addProperty(propertyName, v);
      })(value);
    }

    /// execution
    void apply(std::string methodName,
               const std::vector<std::pair<std::string, PresetValue>> &properties) {
      auto &node = *get_method_node(methodName);
      for (const auto &prop : properties)
        match([&node, &propName = prop.first](const auto &v) { node.setProperty(propName, v); })(
            prop.second);
      node();
    }

  }  // namespace script

}  // namespace zs