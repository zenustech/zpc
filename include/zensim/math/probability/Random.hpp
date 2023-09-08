#pragma once
#include "zensim/TypeAlias.hpp"

/// ref: C++ Weekly With Jason Turner - Episode 44 : constexpr Compile Time Random
namespace zs {

  constexpr u64 seed() noexcept {
    u64 shifted = 0;
    for (const auto c : __TIME__) {
      shifted <<= 8;
      shifted |= c;
    }
    return shifted;
  }

  struct PCG {
    struct pcg32_random_t {
      u64 state = 0;
      u64 inc = seed();
    } rng{};

    using result_type = u32;

    constexpr result_type pcg32_random_r() noexcept {
      u64 oldstate = rng.state;
      rng.state = oldstate * 6364136223846793005ULL + (rng.inc | 1);
      u32 xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
      u32 rot = oldstate >> 59u;
      return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
    static constexpr result_type pcg32_random_r(u64 &state,
                                                u64 inc = 1442695040888963407ull) noexcept {
      u64 oldstate = state;
      state = oldstate * 6364136223846793005ULL + (inc | 1);
      u32 xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
      u32 rot = oldstate >> 59u;
      return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
    // TODO: https://en.wikipedia.org/wiki/Linear_congruential_generator

    constexpr result_type operator()() noexcept { return pcg32_random_r(); }
  };

  constexpr auto call_random(int i = 0) noexcept {
    PCG pcg{};
    while (i > 0) {
      pcg();
      --i;
    }
    return pcg();
  }

}  // namespace zs