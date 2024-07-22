#pragma once
// from taichi/common/core.h

///
/// @brief OS specific macros
///

/// @note Windows
#if defined(_WIN64)
#  define ZS_PLATFORM_WINDOWS
// ref vcruntime.h
#  ifndef ZPC_ACRTIMP
#    if defined _CRTIMP && !defined _VCRT_DEFINED_CRTIMP
#      define ZPC_ACRTIMP _CRTIMP
#    elif !defined _CORECRT_BUILD && defined _DLL
#      define ZPC_ACRTIMP __declspec(dllimport)
#    else
#      define ZPC_ACRTIMP
#    endif
#  endif
#endif

#if defined(_WIN32) && !defined(_WIN64)
static_assert(false, "32-bit Windows systems are not supported")
#endif

#if defined(ZS_PLATFORM_WINDOWS)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif

#  if defined(_DLL) && ZS_BUILD_SHARED_LIBS && !defined(ZS_BUILD_DLL)
#    define ZS_BUILD_DLL
#  endif

#endif

/// @note Android
#if defined(__ANDROID__)
#  define ZS_PLATFORM_ANDROID
#endif

/// @note Linux
#if defined(__linux__) && !defined(ZS_PLATFORM_ANDROID)
#  define ZS_PLATFORM_LINUX
#endif

/// @note OSX
#if defined(__APPLE__)
#  define ZS_PLATFORM_OSX
#endif

/// @note Unix
#if (defined(ZS_PLATFORM_LINUX) || defined(ZS_PLATFORM_OSX))
#  define ZS_PLATFORM_UNIX
#endif

    ///
    /// @brief architecture specific macros
    ///
    /// @note See
    /// https://github.com/abseil/abseil.github.io/blob/master/docs/cpp/platforms/macros.md
    // #if defined(__x86_64__) or defined(_M_X64)
    // #  define ZS_ARCH_AMD64
    // #elif defined(__aarch64__) or defined(_M_ARM64)
    // #  define ZS_ARCH_ARM64
    // #else
    // #endif
    static_assert(sizeof(void *) == 8, "Only 64-bit x86/arm archs are supported.");

///
/// @brief Compiler predefined macros
/// @note See https://sourceforge.net/p/predef/wiki/Compilers/, https://blog.kowalczyk.info/article/j/guide-to-predefined-macros-in-c-compilers-gcc-clang-msvc-etc..html
///

/// @note SYCL_LANGUAGE_VERSION is defined for all sycl compilers
/// @note See https://www.intel.com/content/www/us/en/developer/articles/technical/use-predefined-macros-for-specific-code-for-intel-dpcpp-compiler-intel-cpp-compiler-intel-cpp-compiler-classic.html
#if defined(SYCL_LANGUAGE_VERSION)
#  define ZS_COMPILER_SYCL_VER SYCL_LANGUAGE_VERSION
#endif
/// @note Query if GNUC extensions are supported
#if defined(__GNUC__)
#  define ZS_COMPILER_GNUC_VER __GNUC__
#endif

/// @note Webassembly
#if defined(__EMSCRIPTEN__)
#  define ZS_COMPILER_EMSCRIPTEN
#endif

/// @note This compiler is deprecated by intel, use icx/icpx instead
/// @note See https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/additional-predefined-macros.html
#if defined(__INTEL_COMPILER)
#  define ZS_COMPILER_INTEL_CLASSIC
#endif

/// @note See https://www.intel.com/content/www/us/en/developer/articles/technical/use-predefined-macros-for-specific-code-for-intel-dpcpp-compiler-intel-cpp-compiler-intel-cpp-compiler-classic.html
#if defined(__INTEL_LLVM_COMPILER)
#  define ZS_COMPILER_INTEL_LLVM
#endif

/// @note See https://stackoverflow.com/questions/38499462/how-to-tell-clang-to-stop-pretending-to-be-other-compilers
#if defined(__GNUC__) && !defined(__llvm__) && !defined(ZS_COMPILER_INTEL_CLASSIC)
#  define ZS_COMPILER_GCC
#endif

#if defined(__clang__)
#  define ZS_COMPILER_CLANG
#  if defined(ZS_PLATFORM_OSX)
#    define ZS_COMPILER_APPLE_CLANG
#  endif
#endif

/// @note See https://stackoverflow.com/questions/77012074/macro-to-check-if-i-am-running-msvc-or-intel-compiler
#if defined(_MSC_VER) && !defined(__llvm__) && !defined(ZS_COMPILER_INTEL_CLASSIC)
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

#if defined(ZS_COMPILER_MSVC)  // && !defined(ZS_COMPILER_CLANG)
#  ifdef ZS_BUILD_DLL
#    define ZPC_EXPORT __declspec(dllexport)
#    define ZPC_IMPORT __declspec(dllimport)
#  else
#    define ZPC_EXPORT
#    define ZPC_IMPORT
#  endif

#elif defined(ZS_COMPILER_GCC)
#  define ZPC_EXPORT __attribute__((visibility("default")))
#  define ZPC_IMPORT __attribute__((visibility("default")))

#elif defined(ZS_COMPILER_CLANG)
#  define ZPC_EXPORT __attribute__((visibility("default")))
#  define ZPC_IMPORT __attribute__((visibility("default")))

#elif defined(ZPC_JIT_MODE)
#  define ZPC_EXPORT
#  define ZPC_IMPORT

#else
#  error "unknown compiler!"
#endif

/// @note for zpc core
#ifdef ZPC_CORE_API
#  undef ZPC_CORE_API
#endif
#ifdef ZPC_CORE_PRIVATE
#  define ZPC_CORE_API ZPC_EXPORT
#else
#  define ZPC_CORE_API ZPC_IMPORT
#endif

#ifdef ZPC_CORE_TEMPLATE_EXPORT
#  undef ZPC_CORE_TEMPLATE_EXPORT
#endif
#ifdef ZPC_CORE_TEMPLATE_IMPORT
#  undef ZPC_CORE_TEMPLATE_IMPORT
#endif

#if defined(ZS_COMPILER_MSVC) && defined(ZS_BUILD_DLL)
    #ifdef ZPC_CORE_PRIVATE
        #define ZPC_CORE_TEMPLATE_EXPORT ZPC_EXPORT
        #define ZPC_CORE_TEMPLATE_IMPORT
    #else
        #define ZPC_CORE_TEMPLATE_EXPORT
        #define ZPC_CORE_TEMPLATE_IMPORT ZPC_IMPORT
    #endif
#else
    #define ZPC_CORE_TEMPLATE_EXPORT
    #define ZPC_CORE_TEMPLATE_IMPORT
#endif

/// @note for individual zpc backends
#ifdef ZPC_BACKEND_API
#  undef ZPC_BACKEND_API
#endif
#ifdef ZPC_BACKEND_PRIVATE
#  define ZPC_BACKEND_API ZPC_EXPORT
#else
#  define ZPC_BACKEND_API ZPC_IMPORT
#endif

#ifdef ZPC_BACKEND_TEMPLATE_EXPORT
#  undef ZPC_BACKEND_TEMPLATE_EXPORT
#endif
#ifdef ZPC_BACKEND_TEMPLATE_IMPORT
#  undef ZPC_BACKEND_TEMPLATE_IMPORT
#endif

#if defined(ZS_COMPILER_MSVC) && defined(ZS_BUILD_DLL)
    #ifdef ZPC_BACKEND_PRIVATE
        #define ZPC_BACKEND_TEMPLATE_EXPORT ZPC_EXPORT
        #define ZPC_BACKEND_TEMPLATE_IMPORT
    #else
        #define ZPC_BACKEND_TEMPLATE_EXPORT
        #define ZPC_BACKEND_TEMPLATE_IMPORT ZPC_IMPORT
    #endif
#else
    #define ZPC_BACKEND_TEMPLATE_EXPORT
    #define ZPC_BACKEND_TEMPLATE_IMPORT
#endif

/// @note for the assembled zpc target
#ifdef ZPC_API
#  undef ZPC_API
#endif
#ifdef ZPC_PRIVATE
#  define ZPC_API ZPC_EXPORT
#else
#  define ZPC_API ZPC_IMPORT
#endif

/// template instantiation symbol resolution
#ifdef ZPC_TEMPLATE_EXPORT
#  undef ZPC_TEMPLATE_EXPORT
#endif
#ifdef ZPC_TEMPLATE_IMPORT
#  undef ZPC_TEMPLATE_IMPORT
#endif

#if defined(ZS_COMPILER_MSVC) && defined(ZS_BUILD_DLL)
    #ifdef ZPC_PRIVATE
        #define ZPC_TEMPLATE_EXPORT ZPC_EXPORT
        #define ZPC_TEMPLATE_IMPORT
    #else
        #define ZPC_TEMPLATE_EXPORT
        #define ZPC_TEMPLATE_IMPORT ZPC_IMPORT
    #endif
#else
    #define ZPC_TEMPLATE_EXPORT
    #define ZPC_TEMPLATE_IMPORT
#endif

#undef ZPC_INSTANTIATE
#undef ZPC_INSTANTIATE_CLASS
#undef ZPC_INSTANTIATE_STRUCT
#define ZPC_INSTANTIATE template ZPC_TEMPLATE_EXPORT
#define ZPC_INSTANTIATE_CLASS template class ZPC_TEMPLATE_EXPORT
#define ZPC_INSTANTIATE_STRUCT template struct ZPC_TEMPLATE_EXPORT

#undef ZPC_FWD_DECL_FUNC
#undef ZPC_FWD_DECL_TEMPLATE_CLASS
#undef ZPC_FWD_DECL_TEMPLATE_STRUCT
#define ZPC_FWD_DECL_FUNC extern template ZPC_TEMPLATE_IMPORT
#define ZPC_FWD_DECL_TEMPLATE_CLASS extern template class ZPC_TEMPLATE_IMPORT
#define ZPC_FWD_DECL_TEMPLATE_STRUCT extern template struct ZPC_TEMPLATE_IMPORT

/// @note for extension utilities built upon the zpc target (i.e. zpctool)
#ifdef ZPC_EXTENSION_API
#  undef ZPC_EXTENSION_API
#endif
#ifdef ZPC_EXTENSION_PRIVATE
#  define ZPC_EXTENSION_API ZPC_EXPORT
#else
#  define ZPC_EXTENSION_API ZPC_IMPORT
#endif

#ifdef ZPC_EXTENSION_TEMPLATE_EXPORT
#  undef ZPC_EXTENSION_TEMPLATE_EXPORT
#endif
#ifdef ZPC_EXTENSION_TEMPLATE_IMPORT
#  undef ZPC_EXTENSION_TEMPLATE_IMPORT
#endif

#if defined(ZS_COMPILER_MSVC) && defined(ZS_BUILD_DLL)
    #ifdef ZPC_EXTENSION_PRIVATE
        #define ZPC_EXTENSION_TEMPLATE_EXPORT ZPC_EXPORT
        #define ZPC_EXTENSION_TEMPLATE_IMPORT
    #else
        #define ZPC_EXTENSION_TEMPLATE_EXPORT
        #define ZPC_EXTENSION_TEMPLATE_IMPORT ZPC_IMPORT
    #endif
#else
    #define ZPC_EXTENSION_TEMPLATE_EXPORT
    #define ZPC_EXTENSION_TEMPLATE_IMPORT
#endif

///
/// @brief Useful compiler attributes
///
#if defined(ZS_COMPILER_MSVC)
#  define ZS_NO_INLINE __declspec(noinline)
#else
#  define ZS_NO_INLINE __attribute__((noinline))
#endif


#if defined(ZS_COMPILER_GCC) || defined(ZS_COMPILER_CLANG) || defined(ZS_COMPILER_INTEL_CLASSIC)
#define ZS_LIKELY(x) (__builtin_expect(static_cast<bool>(x), true))
#define ZS_UNLIKELY(x) (__builtin_expect(static_cast<bool>(x), false))

#else
#define ZS_LIKELY(x) (x)
#define ZS_UNLIKELY(x) (x)
#endif

#if defined(ZS_PLATFORM_WINDOWS)
#  define ZS_UNREACHABLE __assume(0);
#else
#  define ZS_UNREACHABLE __builtin_unreachable();
#endif