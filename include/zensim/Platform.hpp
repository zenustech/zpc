#pragma once
// from taichi/common/core.h

// Windows
#if defined(_WIN64)
#  define ZS_PLATFORM_WINDOWS
#endif

#if defined(_WIN32) && !defined(_WIN64)
static_assert(false, "32-bit Windows systems are not supported")
#endif

// Linux
#if defined(__linux__)
#  define ZS_PLATFORM_LINUX
#endif

// OSX
#if defined(__APPLE__)
#  define ZS_PLATFORM_OSX
#endif

// Unix
#if (defined(ZS_PLATFORM_LINUX) || defined(ZS_PLATFORM_OSX))
#  define ZS_PLATFORM_UNIX
#endif

#if defined(ZS_PLATFORM_WINDOWS)
#  define ZS_UNREACHABLE __assume(0);
#else
#  define ZS_UNREACHABLE __builtin_unreachable();
#endif

/// compiler
#if defined(SYCL_LANGUAGE_VERSION)
#  define ZS_COMPILER_SYCL
#endif

#if defined(__INTEL_COMPILER)
#  define ZS_COMPILER_INTEL_CLASSIC
#endif

#if defined(__GNUC__)
#  define ZS_COMPILER_GCC
#endif

#if defined(__clang__)
#  define ZS_COMPILER_CLANG
#endif

#if defined(_MSC_VER)
#  define ZS_COMPILER_MSVC
#endif

#if defined(__NVCC__)
#  define ZS_COMPILER_NVCC
#endif

/// symbol usage/export
// ref: openvdb/Platform.h
#ifdef ZPC_EXPORT
#  undef ZPC_EXPORT
#endif
#ifdef ZPC_IMPORT
#  undef ZPC_IMPORT
#endif
#ifdef ZS_COMPILER_GCC
#  define ZPC_EXPORT __attribute__((visibility("default")))
#  define ZPC_IMPORT __attribute__((visibility("default")))
#endif
#ifdef ZS_COMPILER_MSVC
#  if ZS_BUILD_SHARED_LIBS
#    define ZPC_EXPORT __declspec(dllexport)
#    define ZPC_IMPORT __declspec(dllimport)
#  else
#    define ZPC_EXPORT
#    define ZPC_IMPORT
#  endif
#endif

#ifdef ZPC_API
#  undef ZPC_API
#endif
#if ZPC_PRIVATE
#  define ZPC_API ZPC_EXPORT
#else
#  define ZPC_API ZPC_IMPORT
#endif