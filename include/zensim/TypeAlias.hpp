#pragma once
#include <cstdint>
#include <memory>

namespace zs {

  using uint = unsigned int;
  using i16 = int16_t;
  using i32 = int32_t;
  using i64 = int64_t;
  using u16 = uint16_t;
  using u32 = uint32_t;
  using u64 = uint64_t;
  using f32 = float;
  using f64 = double;

  using dat32 = union {
    f32 f;
    i32 i;
    u32 u;
  };
  using dat64 = union {
    f64 d;
    i64 l;
    u64 ul;
  };

  // kokkos::ObservingRawPtr<T>, OptionalRef<T>
  // vsg::ref_ptr<T>
  template <typename T> using RefPtr = std::decay_t<T> *;             ///< non-owning reference
  template <typename T> using ConstRefPtr = const std::decay_t<T> *;  ///< non-owning reference
  template <typename T> using Holder = std::unique_ptr<T>;            ///< non-owning reference

  using NodeID = i32;
  using ProcID = char;
  using StreamID = u32;
  using EventID = u32;

/// lambda capture
/// https://vittorioromeo.info/index/blog/capturing_perfectly_forwarded_objects_in_lambdas.html
#define FWD(...) ::std::forward<decltype(__VA_ARGS__)>(__VA_ARGS__)

}  // namespace zs
