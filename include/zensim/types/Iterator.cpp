#include "Iterator.h"

namespace zs {

#if 0
  template <> input_iterator_tag::operator std::input_iterator_tag() { return {}; }
  template <> output_iterator_tag::operator std::output_iterator_tag() { return {}; }
  template <> forward_iterator_tag::operator std::forward_iterator_tag() { return {}; }
  template <> bidirectional_iterator_tag::operator std::bidirectional_iterator_tag() { return {}; }
  template <> random_access_iterator_tag::operator std::random_access_iterator_tag() { return {}; }
  // template<>
  // contiguous_iterator_tag::operator std::contiguous_iterator_tag() {return {};}
#endif

}  // namespace zs