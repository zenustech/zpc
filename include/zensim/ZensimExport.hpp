#pragma once

// https://abseil.io/docs/cpp/platforms/macros

#ifndef ZENSIM_EXPORT
#  if defined(_MSC_VER) || defined(__CYGWIN__)
#    ifdef Zensim_EXPORT
#      define ZENSIM_EXPORT __declspec(dllexport)
#    else
#      define ZENSIM_EXPORT __declspec(dllimport)
#    endif
#  elif defined(__clang__) || defined(__GNUC__)
#    define ZENSIM_EXPORT __attribute__((visibility("default")))
#  endif
#endif
