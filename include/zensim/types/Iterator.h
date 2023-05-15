#pragma once
#include <functional>
#include <iterator>
#include <limits>
#include <type_traits>

#include "zensim/ZpcIterator.hpp"

namespace zs {

#if 0
  extern template input_iterator_tag::operator std::input_iterator_tag();
  extern template output_iterator_tag::operator std::output_iterator_tag();
  extern template forward_iterator_tag::operator std::forward_iterator_tag();
  extern template bidirectional_iterator_tag::operator std::bidirectional_iterator_tag();
  extern template random_access_iterator_tag::operator std::random_access_iterator_tag();
#endif

  template <typename Iter>
  struct iterator_traits<Iter, void_t<typename std::iterator_traits<Iter>::reference,
                                      typename std::iterator_traits<Iter>::pointer,
                                      typename std::iterator_traits<Iter>::difference_type,
                                      typename std::iterator_traits<Iter>::value_type,
                                      typename std::iterator_traits<Iter>::iterator_category,
                                      enable_if_t<!is_base_of_v<IteratorInterface<Iter>, Iter>>>> {
    // reference
    using reference = typename std::iterator_traits<Iter>::reference;
    // pointer
    using pointer = typename std::iterator_traits<Iter>::pointer;
    // difference type
    using difference_type = typename std::iterator_traits<Iter>::difference_type;
    // value type
    using value_type = typename std::iterator_traits<Iter>::value_type;
    // iterator category
    using iterator_category = typename std::iterator_traits<Iter>::iterator_category;
  };

}  // namespace zs