// ref: openvdb NodeMask
// see https://github.com/AcademySoftwareFoundation/openvdb/blob/master/LICENSE
#pragma once
#include "zensim/TypeAlias.hpp"
#include "zensim/execution/Atomics.hpp"
#include "zensim/execution/Intrinsics.hpp"

namespace zs {

  /// @brief mask type for at least Size bits
  template <int Size> struct bit_mask {
    using word_type = conditional_t<sizeof(void*) == 8, u64, u32>;
    static_assert(sizeof(word_type) == sizeof(void*),
                  "word_type should be the same size as a pointer");
    static constexpr int bits_per_word = sizeof(word_type) * 8;
    static_assert((bits_per_word & (bits_per_word - 1)) == 0,
                  "bits_per_word should be power of two");
    static constexpr int word_mask = bits_per_word - 1;
    static constexpr int log_2_word_size = bit_count(bits_per_word);
    static constexpr int bit_size = Size;
    static constexpr int word_count = (bit_size + bits_per_word - 1) / bits_per_word;
    static_assert(word_count > 0, "word count should be positive");

    word_type words[word_count];  // only member data!

  public:
    constexpr bit_mask() noexcept { setOff(); }
    constexpr bit_mask(bool on) { set(on); }
    constexpr bit_mask(const bit_mask& other) = default;
    constexpr bit_mask(bit_mask&& other) noexcept = default;
    ~bit_mask() = default;
    /// Assignment operator
    constexpr bit_mask& operator=(const bit_mask& other) = default;
    constexpr bit_mask& operator=(bit_mask&& other) noexcept = default;

    constexpr bool operator==(const bit_mask& other) const {
      int n = word_count;
      for (const word_type *w1 = words, *w2 = other.words; n-- && *w1++ == *w2++;)
        ;
      return n == -1;
    }

    constexpr bool operator!=(const bit_mask& other) const { return !(*this == other); }

    /// @brief Bitwise intersection
    constexpr bit_mask& operator&=(const bit_mask& other) {
      word_type* w1 = words;
      const word_type* w2 = other.words;
      for (int n = word_count; n--; ++w1, ++w2) *w1 &= *w2;
      return *this;
    }
    /// @brief Bitwise union
    constexpr bit_mask& operator|=(const bit_mask& other) {
      word_type* w1 = words;
      const word_type* w2 = other.words;
      for (int n = word_count; n--; ++w1, ++w2) *w1 |= *w2;
      return *this;
    }
    /// @brief Bitwise difference
    constexpr bit_mask& operator-=(const bit_mask& other) {
      word_type* w1 = words;
      const word_type* w2 = other.words;
      for (int n = word_count; n--; ++w1, ++w2) *w1 &= ~*w2;
      return *this;
    }
    /// @brief Bitwise XOR
    constexpr bit_mask& operator^=(const bit_mask& other) {
      word_type* w1 = words;
      const word_type* w2 = other.words;
      for (int n = word_count; n--; ++w1, ++w2) *w1 ^= *w2;
      return *this;
    }
    constexpr bit_mask operator!() const {
      bit_mask m(*this);
      m.toggle();
      return m;
    }
    constexpr bit_mask operator&(const bit_mask& other) const {
      bit_mask m(*this);
      m &= other;
      return m;
    }
    constexpr bit_mask operator|(const bit_mask& other) const {
      bit_mask m(*this);
      m |= other;
      return m;
    }
    constexpr bit_mask operator^(const bit_mask& other) const {
      bit_mask m(*this);
      m ^= other;
      return m;
    }

    /// Return the byte size of this bit_mask
    static constexpr int memUsage() { return static_cast<int>(word_count * sizeof(word_type)); }
    /// Return the total number of on bits
    template <execspace_e space = deduce_execution_space()>
    constexpr int countOn(wrapv<space> tag = {}) const {
      int sum = 0, n = word_count;
      for (const word_type* w = words; n--; ++w) sum += count_ones(*w, tag);
      return sum;
    }
    /// Return the total number of on bits
    template <execspace_e space = deduce_execution_space()>
    constexpr int countOffset(int k, wrapv<space> tag = {}) const {
      int sum = 0, n = k >> log_2_word_size;
      const word_type* w = words;
      for (; n--; ++w) sum += count_ones(*w, tag);
      sum += count_ones((*w) & (((word_type)1 << (word_type)(k & word_mask)) - 1), tag);
      return sum;
    }
    /// Return the total number of on bits
    template <execspace_e space = deduce_execution_space()>
    constexpr int countOff(wrapv<space> tag = {}) const {
      return bit_size - countOn(tag);
    }
    /// Set the <i>n</i>th  bit on
    constexpr void setOn(int n) {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (n >= bit_size) printf("[%d]-th bit is out of bound [%d]\n", n, bit_size);
#endif
      words[n >> log_2_word_size] |= word_type(1) << (n & word_mask);
    }
    template <execspace_e space = deduce_execution_space()>
    constexpr void setOn(int n, wrapv<space>) {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (n >= bit_size) printf("[%d]-th bit is out of bound [%d]\n", n, bit_size);
#endif
      atomic_or(wrapv<space>{}, &words[n >> log_2_word_size], word_type(1) << (n & word_mask));
      // words[n >> log_2_word_size] |= word_type(1) << (n & word_mask);
    }
    /// Set the <i>n</i>th bit off
    constexpr void setOff(int n) {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (n >= bit_size) printf("[%d]-th bit is out of bound [%d]\n", n, bit_size);
#endif
      words[n >> log_2_word_size] &= ~(word_type(1) << (n & word_mask));
    }
    template <execspace_e space = deduce_execution_space()>
    constexpr void setOff(int n, wrapv<space>) {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (n >= bit_size) printf("[%d]-th bit is out of bound [%d]\n", n, bit_size);
#endif
      atomic_and(wrapv<space>{}, &words[n >> log_2_word_size], ~(word_type(1) << (n & word_mask)));
      // words[n >> log_2_word_size] &= ~(word_type(1) << (n & word_mask));
    }
    /// Set the <i>n</i>th bit to the specified state
    constexpr void set(int n, bool On) { (void)(On ? setOn(n) : setOff(n)); }
    /// Set all bits to the specified state
    constexpr void set(bool on) {
      const word_type state = on ? ~word_type(0) : word_type(0);
      int n = word_count;
      for (word_type* w = words; n--; ++w) *w = state;
    }
    /// Set all bits on
    constexpr void setOn() {
      int n = word_count;
      for (word_type* w = words; n--; ++w) *w = ~word_type(0);
    }
    /// Set all bits off
    constexpr void setOff() {
      int n = word_count;
      for (word_type* w = words; n--; ++w) *w = word_type(0);
    }
    /// Toggle the state of the <i>n</i>th bit
    constexpr void toggle(int n) {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      // if ((n >> log_2_word_size) >= word_count)
      if (n >= bit_size) printf("[%d]-th bit is out of bound [%d]\n", n, bit_size);
#endif
      words[n >> log_2_word_size] ^= word_type(1) << (n & word_mask);
    }
    /// Toggle the state of all bits in the mask
    constexpr void toggle() {
      int n = word_count;
      for (word_type* w = words; n--; ++w) *w = ~*w;
    }
    /// Set the first bit on
    constexpr void setFirstOn() { setOn(0); }
    /// Set the last bit on
    constexpr void setLastOn() { setOn(bit_size - 1); }
    /// Set the first bit off
    constexpr void setFirstOff() { setOff(0); }
    /// Set the last bit off
    constexpr void setLastOff() { setOff(bit_size - 1); }
    /// Return @c true if the <i>n</i>th bit is on
    constexpr bool isOn(int n) const {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (n >= bit_size) printf("[%d]-th bit is out of bound [%d]\n", n, bit_size);
#endif
      return 0 != (words[n >> log_2_word_size] & (word_type(1) << (n & word_mask)));
    }
    /// Return @c true if the <i>n</i>th bit is off
    constexpr bool isOff(int n) const { return !isOn(n); }
    /// Return @c true if all the bits are on
    constexpr bool isOn() const {
      int n = word_count;
      for (const word_type* w = words; n-- && *w++ == ~word_type(0);)
        ;
      return n == -1;
    }
    /// Return @c true if all the bits are off
    constexpr bool isOff() const {
      int n = word_count;
      for (const word_type* w = words; n-- && *w++ == word_type(0);)
        ;
      return n == -1;
    }
    /// Return @c true if bits are either all off OR all on.
    /// @param isOn Takes on the values of all bits if the method
    /// returns true - else it is undefined.
    constexpr bool isConstant(bool& isOn) const {
      isOn = (words[0] == ~word_type(0));                   // first word has all bits on
      if (!isOn && words[0] != word_type(0)) return false;  // early out
      const word_type *w = words + 1, *n = words + word_count;
      while (w < n && *w == words[0]) ++w;
      return w == n;
    }
    template <execspace_e space = deduce_execution_space()>
    constexpr int findFirstOn(wrapv<space> tag = {}) const {
      int n = 0;
      const word_type* w = words;
      for (; n < word_count && !*w; ++w, ++n)
        ;
      return n == word_count ? bit_size : (n << log_2_word_size) + count_tailing_zeros(*w, tag);
    }
    template <execspace_e space = deduce_execution_space()>
    constexpr int findFirstOff(wrapv<space> tag = {}) const {
      int n = 0;
      const word_type* w = words;
      for (; n < word_count && !~*w; ++w, ++n)
        ;
      return n == word_count ? bit_size : (n << log_2_word_size) + count_tailing_zeros(~*w, tag);
    }

    //@{
    /// Return the <i>n</i>th word of the bit mask, for a word of arbitrary size.
    template <typename WordT> constexpr WordT getWord(int n) const {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (n * 8 * sizeof(WordT) >= bit_size)
        printf("[%d]-th word is out of bound [%d]\n", n, bit_size);
#endif
      return reinterpret_cast<const WordT*>(words)[n];
    }
    template <typename WordT> constexpr WordT& getWord(int n) {
#if ZS_ENABLE_OFB_ACCESS_CHECK
      if (n * 8 * sizeof(WordT) >= bit_size)
        printf("[%d]-th word is out of bound [%d]\n", n, bit_size);
#endif
      return reinterpret_cast<WordT*>(words)[n];
    }
    //@}

    template <execspace_e space = deduce_execution_space()>
    constexpr int findNextOn(int start, wrapv<space> tag = {}) const {
      int n = start >> log_2_word_size;      // initiate
      if (n >= word_count) return bit_size;  // check for out of bounds
      int m = start & word_mask;
      word_type b = words[n];
      if (b & (word_type(1) << m)) return start;    // simpel case: start is on
      b &= ~word_type(0) << m;                      // mask out lower bits
      while (!b && ++n < word_count) b = words[n];  // find next none-zero word
      return (!b ? bit_size
                 : (n << log_2_word_size) + count_tailing_zeros(b, tag));  // catch last word=0
    }

    template <execspace_e space = deduce_execution_space()>
    constexpr int findNextOff(int start, wrapv<space> tag = {}) const {
      int n = start >> log_2_word_size;      // initiate
      if (n >= word_count) return bit_size;  // check for out of bounds
      int m = start & word_mask;
      word_type b = ~words[n];
      if (b & (word_type(1) << m)) return start;     // simpel case: start is on
      b &= ~word_type(0) << m;                       // mask out lower bits
      while (!b && ++n < word_count) b = ~words[n];  // find next none-zero word
      return (!b ? bit_size
                 : (n << log_2_word_size) + count_tailing_zeros(b, tag));  // catch last word=0
    }
  };

}  // namespace zs