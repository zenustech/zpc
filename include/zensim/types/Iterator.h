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

  template <typename Iter> struct iterator_traits<
      Iter, enable_if_type<!is_base_of_v<IteratorInterface<Iter>, Iter>,
                           void_t<typename std::iterator_traits<Iter>::reference,
                                  typename std::iterator_traits<Iter>::pointer,
                                  typename std::iterator_traits<Iter>::difference_type,
                                  typename std::iterator_traits<Iter>::value_type,
                                  typename std::iterator_traits<Iter>::iterator_category>>> {
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

namespace std {

  template <typename Iter> struct iterator_traits<zs::LegacyIterator<Iter>> {
    // reference
    using reference = typename zs::iterator_traits<Iter>::reference;
    // pointer
    using pointer = typename zs::iterator_traits<Iter>::pointer;
    // difference type
    using difference_type = typename zs::iterator_traits<Iter>::difference_type;
    // value type
    using value_type = typename zs::iterator_traits<Iter>::value_type;
    // iterator category
    using zs_iterator_category = typename zs::iterator_traits<Iter>::iterator_category;
    using iterator_category = conditional_t<
        is_same_v<zs_iterator_category, zs::input_iterator_tag>, std::input_iterator_tag,
        conditional_t<
            is_same_v<zs_iterator_category, zs::output_iterator_tag>, std::output_iterator_tag,
            conditional_t<
                is_same_v<zs_iterator_category, zs::forward_iterator_tag>,
                std::forward_iterator_tag,
                conditional_t<
                    is_same_v<zs_iterator_category, zs::bidirectional_iterator_tag>,
                    std::bidirectional_iterator_tag,
                    conditional_t<is_same_v<zs_iterator_category, zs::random_access_iterator_tag>,
                                  std::random_access_iterator_tag,
                                  std::random_access_iterator_tag>>>>>;
  };

}  // namespace std