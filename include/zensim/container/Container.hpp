#pragma once
#include <zensim/types/Object.h>

#include <zensim/types/RuntimeStructurals.hpp>

namespace zs {

  /// using Value = variant<char, i32, float, i64, double, vec2, vec3, vec4, mat2,
  /// mat3, mat4>;

  /// https://stackoverflow.com/questions/41878040/does-boostany-stdany-store-small-objects-in-place
  /// Value operator()(std::any)
  /// Value operator()(char, std::any)
  /// Value operator()(wrapv<I>, char, std::any)
  struct ContainerStructure {
    virtual ~ContainerStructure() {}
    // virtual ;
  };
  template <typename Instance> struct SpecificStructure : ContainerStructure {
    template <typename SnodeInst> SpecificStructure(SnodeInst &&si)
        : _inst{std::forward<SnodeInst>(si)} {}

  private:
    Instance _inst;
  };

  struct Container;
  ZS_type_name(Container);
  struct Container : Inherit<Object, Container> {
    /// iterator/ begin/ end
    std::unique_ptr<ContainerStructure> _struct;
    // MemoryPolicy _mem;
  };

}  // namespace zs