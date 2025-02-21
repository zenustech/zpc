cmake_minimum_required(VERSION 3.18) # support relative path, cuda_std_17, archive manipulation

message("===============================================================")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug")
endif()

# project meta
# ============
include("project_info.in")
project(${project_name}
  VERSION ${project_version}
  DESCRIPTION ${project_description}
  LANGUAGES C CXX)

set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_BINARY_DIR})

# global rpath setup (before target creations)
# ============================================
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling#common-questions
# https://cmake.org/pipermail/cmake/2008-January/019290.html
if(NOT APPLE)
  set(CMAKE_INSTALL_RPATH $ORIGIN/../${CMAKE_INSTALL_LIBDIR}) # location of the binary requiring the dep
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}" isInstallRpathSystemDir)

if("${isInstallRpathSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif("${isInstallRpathSystemDir}" STREQUAL "-1")

#[=[
file(RELATIVE_PATH relDir
  ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
  ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH $ORIGIN $ORIGIN/${relDir})
]=]

# build options
# =============
option(CMAKE_EXPORT_COMPILE_COMMANDS "Display Compile Commands" 1)
# general cmake build setup
option(ZS_ENABLE_PCH "Enable Precompiled Headers" OFF)
option(ZS_ENABLE_TEST "Enable Tests" OFF)
option(ZS_ENABLE_DOC "Enable Doxygen Documentation Generation" OFF)
option(ZS_BUILD_SHARED_LIBS "Enable compilation of shared libraries" OFF)
option(ZS_PROPAGATE_DEPS "Pass on dependencies (TBD)" ON)
# debug
option(ZS_ENABLE_OFB_ACCESS_CHECK "Enable out-of-bound access check" OFF)
# cuda
option(ZS_ENABLE_CUDA "Enable cuda backend" ON)
option(ZS_ENABLE_ZENO_CU_WRANGLE "Enable cuda wrangles for zeno" OFF)
option(ZS_AUTO_DETECT_CUDA_ARCH "Let cmake automatically determine target cuda architecture" ON)
# musa
option(ZS_ENABLE_MUSA "Enable musa backend" ON)
option(ZS_AUTO_DETECT_MUSA_ARCH "Let cmake automatically determine target musa architecture" ON)
# vulkan
option(ZS_ENABLE_VULKAN "Enable vulkan backend" ON)
# openmp
option(ZS_ENABLE_OPENMP "Enable openmp backend" ON)
# JIT (llvm/clang)
option(ZS_ENABLE_JIT "Enable JIT module" ON)
# sycl, cl
option(ZS_ENABLE_SYCL_ONEAPI "Enable SYCL[Intel-OneAPI] backend" ON)
option(ZS_ENABLE_SYCL_ACPP "Enable SYCL[AdaptiveCpp (originally hipSYCL)] backend" ON)
option(ZS_ENABLE_OPENCL "Enable OpenCL backend" OFF)
# python
option(ZS_ENABLE_PYTHON "Enable python-related build" ON)
set(ZS_OVERWRITE_PYTHON_INCLUDE_DIR "" CACHE STRING "User overwriting of python include directory")
set(ZS_OVERWRITE_PYTHON_LIBRARIES "" CACHE STRING "User overwriting of python libraries")
set(ZS_OVERWRITE_PYTHON_EXECUTABLE "" CACHE STRING "User overwriting of python executable")
# serialization
option(ZS_ENABLE_SERIALIZATION "Enable ZPC serialization" OFF)
option(ZS_SILENT_ABSENCE_OF_SERIALIZATION_IMPL "No throw upon the absence of ZPC serialization implementation" ON)
# openvdb
option(ZS_ENABLE_OPENVDB "Enable openvdb-related build" ON)
# package/shipping
option(ZS_ENABLE_INSTALL "Install targets" OFF)
option(ZS_ENABLE_PACKAGE "Build package" OFF)

# option(ZENSIM_CORE_SHARED "Build dynamically linked version of the core library." ON)
# option(ZENSIM_CORE_STATIC "Build statically linked version of the core library." ON)

if(CMAKE_VERSION VERSION_LESS "3.21")
  # ref: VulkanMemoryAllocator repo
  # https://cmake.org/cmake/help/latest/variable/PROJECT_IS_TOP_LEVEL.html
  string(COMPARE EQUAL ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR} PROJECT_IS_TOP_LEVEL)
endif()

if(PROJECT_IS_TOP_LEVEL)
  if(ZS_ENABLE_INSTALL)
    set(ZS_ENABLE_INSTALL OFF)
  endif(ZS_ENABLE_INSTALL)
endif(PROJECT_IS_TOP_LEVEL)

# project base targets
# ====================
include(CMake/Utility.cmake)

# include(CMake/ClangSupport.cmake)
include(CMake/Cache.cmake)
include(CMake/StaticAnalyzers.cmake)
include(CMake/Sanitizers.cmake)
include(CMake/CompilerWarnings.cmake)

# ---- dependencies ----
# ======================
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  if(ZS_ENABLE_CUDA)
    set(ZS_ENABLE_CUDA OFF)
    message(STATUS "== Intel compiler as host compiler is not supported by NVidia nvcc, thus turned off. ==")
  endif(ZS_ENABLE_CUDA)
endif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")

add_subdirectory(include/zensim/zpc_tpls) # build zsproj_deps target
enable_sanitizers(zsproj_deps)

if(${ZS_ENABLE_SYCL_ONEAPI} OR ${ZS_ENABLE_SYCL_ACPP})
  set(ZS_ENABLE_PCH OFF)
  message(STATUS "== When SYCL is enabled, precompile-headers are disabled. ==")
endif(${ZS_ENABLE_SYCL_ONEAPI} OR ${ZS_ENABLE_SYCL_ACPP})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(ZS_ENABLE_VULKAN)
    set(ZS_ENABLE_VULKAN_VALIDATION ON)
  endif(ZS_ENABLE_VULKAN)
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

if(ZS_ENABLE_PCH)
  # This sets a global PCH parameter, each project will build its own PCH, which
  # is a good idea if any #define's change
  # consider breaking this out per project as necessary
  target_precompile_headers(zsproj_deps INTERFACE <vector> <string> <map> <utility> <cstdio>)
endif()

# ---- build setup -----
# ========cuda==========
if(ZS_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)

  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message("-- cuda-compiler " ${CMAKE_CUDA_COMPILER})
  else(CMAKE_CUDA_COMPILER)
    message(STATUS "No CUDA support")
    set(ZS_ENABLE_CUDA OFF)
  endif(CMAKE_CUDA_COMPILER)

  set(CUDA_FOUND ${CMAKE_CUDA_COMPILER})
endif(ZS_ENABLE_CUDA)

# ========proj==========
if(MSVC)
  target_compile_definitions(zsproj_deps INTERFACE NOMINMAX)
endif()

target_include_directories(zsproj_deps INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(zsproj_deps INTERFACE
  $<BUILD_INTERFACE:ZS_INCLUDE_DIR="${PROJECT_SOURCE_DIR}/include">
  $<INSTALL_INTERFACE:ZS_INCLUDE_DIR="${CMAKE_INSTALL_INCLUDEDIR}">
)

# ------ cxx ------
# =================
add_library(zpc_cxx_deps INTERFACE)
if(${CMAKE_CXX_COMPILER_ID} MATCHES "AppleClang")
  target_compile_options(zpc_cxx_deps INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>: -Wno-deprecated-builtins>
  )
else()
  # target_compile_features(zpc_cxx_deps INTERFACE cxx_std_17)
endif()
target_compile_features(zpc_cxx_deps INTERFACE cxx_std_17)
target_link_libraries(zpc_cxx_deps INTERFACE zsproj_deps)
if(WIN32)
  target_compile_definitions(zpc_cxx_deps
    INTERFACE -DWIN32_LEAN_AND_MEAN
  )
endif(WIN32)

# out of boundary check
if(ZS_ENABLE_OFB_ACCESS_CHECK)
  target_compile_definitions(zpc_cxx_deps
    INTERFACE ZS_ENABLE_OFB_ACCESS_CHECK=1
  )
else(ZS_ENABLE_OFB_ACCESS_CHECK)
  target_compile_definitions(zpc_cxx_deps
    INTERFACE ZS_ENABLE_OFB_ACCESS_CHECK=0
  )
endif(ZS_ENABLE_OFB_ACCESS_CHECK)

if(ZS_ENABLE_VULKAN_VALIDATION)
  target_compile_definitions(zpc_cxx_deps
    INTERFACE ZS_ENABLE_VULKAN_VALIDATION=1
  )
else(ZS_ENABLE_VULKAN_VALIDATION)
  target_compile_definitions(zpc_cxx_deps
    INTERFACE ZS_ENABLE_VULKAN_VALIDATION=0
  )
endif(ZS_ENABLE_VULKAN_VALIDATION)

# serialization
if(ZS_ENABLE_SERIALIZATION)
  target_compile_definitions(zpc_cxx_deps
    INTERFACE ZS_ENABLE_SERIALIZATION=1
  )

  if(ZS_SILENT_ABSENCE_OF_SERIALIZATION_IMPL)
    target_compile_definitions(zpc_cxx_deps
      INTERFACE ZS_SILENT_ABSENCE_OF_SERIALIZATION_IMPL=1
    )
  else(ZS_SILENT_ABSENCE_OF_SERIALIZATION_IMPL)
    target_compile_definitions(zpc_cxx_deps
      INTERFACE ZS_SILENT_ABSENCE_OF_SERIALIZATION_IMPL=0
    )
  endif(ZS_SILENT_ABSENCE_OF_SERIALIZATION_IMPL)

else(ZS_ENABLE_SERIALIZATION)
  target_compile_definitions(zpc_cxx_deps
    INTERFACE ZS_ENABLE_SERIALIZATION=0
    ZS_SILENT_ABSENCE_OF_SERIALIZATION_IMPL=0
  )
endif(ZS_ENABLE_SERIALIZATION)

if(ZS_ENABLE_INSTALL)
  if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(RESOURCE_BASE_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(RESOURCE_AT_RELATIVE_PATH 0)
  elseif(DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(RESOURCE_BASE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(RESOURCE_AT_RELATIVE_PATH 1)
  else()
    set(RESOURCE_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR})
    set(RESOURCE_AT_RELATIVE_PATH 0)
  endif()

else(ZS_ENABLE_INSTALL)
  if(DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(RESOURCE_BASE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(RESOURCE_AT_RELATIVE_PATH 1)
  elseif(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(RESOURCE_BASE_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(RESOURCE_AT_RELATIVE_PATH 0)
  else()
    set(RESOURCE_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR})
    set(RESOURCE_AT_RELATIVE_PATH 0)
  endif()

endif(ZS_ENABLE_INSTALL)

target_compile_definitions(zpc_cxx_deps
  # INTERFACE AssetDirPath="${PROJECT_SOURCE_DIR}/zpc_assets"
  INTERFACE AssetDirPath="${RESOURCE_BASE_DIR}"
  RESOURCE_AT_RELATIVE_PATH=${RESOURCE_AT_RELATIVE_PATH}
)
message("[AssetDirPath] set to: [${RESOURCE_BASE_DIR}]")
message("[RESOURCE_AT_RELATIVE_PATH] set to: ${RESOURCE_AT_RELATIVE_PATH}")

if(ZS_BUILD_SHARED_LIBS)
  target_compile_definitions(zpc_cxx_deps
    INTERFACE ZS_BUILD_SHARED_LIBS=1
  )
else()
  target_compile_definitions(zpc_cxx_deps
    INTERFACE ZS_BUILD_SHARED_LIBS=0
  )
endif()

target_link_libraries(zpc_deps INTERFACE zpc_cxx_deps)

# ---- binaries ----
# ==================

# ---- library ----
# =================
add_subdirectory(include/zensim)

if(NOT PROJECT_IS_TOP_LEVEL)
  set(ZS_ENABLE_OPENGL ${ZS_ENABLE_OPENGL} PARENT_SCOPE)

  # backends
  set(ZS_ENABLE_CUDA ${ZS_ENABLE_CUDA} PARENT_SCOPE)
  set(ZS_ENABLE_VULKAN ${ZS_ENABLE_VULKAN} PARENT_SCOPE)
  set(ZS_ENABLE_OPENMP ${ZS_ENABLE_OPENMP} PARENT_SCOPE)
  set(ZS_ENABLE_JIT ${ZS_ENABLE_JIT} PARENT_SCOPE)
  set(ZS_ENABLE_SYCL_ONEAPI ${ZS_ENABLE_SYCL_ONEAPI} PARENT_SCOPE)
  set(ZS_ENABLE_SYCL_ACPP ${ZS_ENABLE_SYCL_ACPP} PARENT_SCOPE)
  set(ZS_ENABLE_OPENCL ${ZS_ENABLE_OPENCL} PARENT_SCOPE)
  set(ZS_ENABLE_PTHREADS ${ZS_ENABLE_PTHREADS} PARENT_SCOPE)
  # 3rd party utils
  set(ZS_ENABLE_SERIALIZATION ${ZS_ENABLE_SERIALIZATION} PARENT_SCOPE)
  set(ZS_ENABLE_OPENVDB ${ZS_ENABLE_OPENVDB} PARENT_SCOPE)
  set(ZS_ENABLE_PARTIO ${ZS_ENABLE_PARTIO} PARENT_SCOPE)
  set(ZS_ENABLE_MSHIO ${ZS_ENABLE_MSHIO} PARENT_SCOPE)

  if(ZS_ENABLE_CUDA)
    # propagate cmake zs variables about cuda
    set(ZS_CUDA_ARCH ${ZS_CUDA_ARCH} PARENT_SCOPE)

    if(ZS_AUTO_DETECT_CUDA_ARCH)
      set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} PARENT_SCOPE)
    endif(ZS_AUTO_DETECT_CUDA_ARCH)
  endif(ZS_ENABLE_CUDA)

  set(ZS_PYTHON_FOUND ${ZS_PYTHON_FOUND} PARENT_SCOPE)
  set(ZS_OVERWRITE_PYTHON_LIBRARIES ${ZS_OVERWRITE_PYTHON_LIBRARIES} PARENT_SCOPE)
  set(ZS_OVERWRITE_PYTHON_INCLUDE_DIR ${ZS_OVERWRITE_PYTHON_INCLUDE_DIR} PARENT_SCOPE)
  set(ZS_OVERWRITE_PYTHON_EXECUTABLE ${ZS_OVERWRITE_PYTHON_EXECUTABLE} PARENT_SCOPE)
  set(ZS_OVERWRITE_PYTHON_LINK_DIRECTORY ${ZS_OVERWRITE_PYTHON_LINK_DIRECTORY} PARENT_SCOPE)
  if(WIN32)
    set(ZS_PYTHON_DLLS ${ZS_PYTHON_DLLS} PARENT_SCOPE)
  endif()
endif(NOT PROJECT_IS_TOP_LEVEL)

message("cmake configure summary")
message("project name: ${project_name}")
message("package version: ${project_version}")
message("backend build options: pthreads: ${ZS_ENABLE_PTHREADS}, sycl(adaptivecpp): ${ZS_ENABLE_SYCL_ACPP}, sycl(oneapi): ${ZS_ENABLE_SYCL_ONEAPI}, cuda: ${ZS_ENABLE_CUDA}(${CUDAToolkit_VERSION}), musa: ${ZS_ENABLE_MUSA}(${MUSA_VERSION}), omp: ${ZS_ENABLE_OPENMP}, vulkan: ${ZS_ENABLE_VULKAN}, cl: ${ZS_ENABLE_OPENCL}, jit(llvm): ${ZS_ENABLE_JIT}.")
message("dependent library options: python: ${ZS_OVERWRITE_PYTHON_EXECUTABLE}, openvdb ${ZS_ENABLE_OPENVDB}, opengl ${ZS_ENABLE_OPENGL}, partio ${ZS_ENABLE_PARTIO}, mshio ${ZS_ENABLE_MSHIO}")

# ----  Docs   ----
# =================
if(ZS_ENABLE_DOC)
  set(DOXYGEN_CALLER_GRAPH YES)
  set(DOXYGEN_CALL_GRAPH YES)
  set(DOXYGEN_EXTRACT_ALL YES)
  find_package(Doxygen)

  if(DOXYGEN_FOUND)
    find_package(Sphinx COMPONENTS breathe)
    message("Setting up document cmake targets")
    set(DOXYGEN_EXCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include/zensim/zpc_tpls)

    if(Sphinx_FOUND)
      set(DOXYGEN_GENERATE_HTML YES)
    endif(Sphinx_FOUND)

    set(DOXYGEN_GENERATE_XML YES)
    doxygen_add_docs(zpc_docs_doxygen include
      COMMENT "Generating ZPC API documentation with Doxygen"
    )

    if(Sphinx_FOUND)
      set(SPHINX_VERSION ${PROJECT_VERSION})
      set(SPHINX_LANGUAGE zh_CN)
      sphinx_add_docs(
        zpc_docs_sphinx
        BREATHE_PROJECTS zpc_docs_doxygen
        BUILDER html
        SOURCE_DIRECTORY docs
      )
    else(Sphinx_FOUND)
      message("Sphinx(with breathe) not found. Skipping target [zpc_docs_sphinx].")
    endif(Sphinx_FOUND)
  else(DOXYGEN_FOUND)
    message("Doxygen not found. Skipping target [zpc_docs_doxygen].")
  endif(DOXYGEN_FOUND)
endif(ZS_ENABLE_DOC)

# ----  Tests  ----
# =================
if(ZS_ENABLE_TEST)
  include(CTest)
  enable_testing()
  add_subdirectory(test)
endif(ZS_ENABLE_TEST)

# ---- Install ----
# =================
if(ZS_ENABLE_INSTALL)
  set(ZS_INSTALL_DEP_TARGETS zspartio zswhereami)

  if(ZS_ENABLE_OPENVDB)
    set(ZS_INSTALL_DEP_TARGETS ${ZS_INSTALL_DEP_TARGETS} zsopenvdb)
  endif(ZS_ENABLE_OPENVDB)

  set(INSTALL_TARGETS zensim zpc zpccore zpctool zpc_cxx_deps zsproj_deps ${ZS_INSTALL_DEP_TARGETS})

  if(ZS_ENABLE_OPENMP)
    set(INSTALL_TARGETS ${INSTALL_TARGETS} zpcomp zpc_omp_deps)
  endif(ZS_ENABLE_OPENMP)

  if(ZS_ENABLE_CUDA)
    set(INSTALL_TARGETS ${INSTALL_TARGETS} zpccuda zpc_cuda_deps)
  endif(ZS_ENABLE_CUDA)

  if(ZS_ENABLE_SYCL_ONEAPI OR ZS_ENABLE_SYCL_ACPP)
    set(INSTALL_TARGETS ${INSTALL_TARGETS} zpcsycl zpc_sycl_deps)
  endif()

  if(ZS_ENABLE_OPENCL)
    set(INSTALL_TARGETS ${INSTALL_TARGETS} zpccl zpc_cl_deps)
  endif(ZS_ENABLE_OPENCL)

  if(ZS_ENABLE_JIT)
    set(INSTALL_TARGETS ${INSTALL_TARGETS} zpc_py_interop zpc_jit_clang)
    if(ZS_ENABLE_CUDA)
      set(INSTALL_TARGETS ${INSTALL_TARGETS} zpc_jit_nvrtc)
    endif(ZS_ENABLE_CUDA)
  endif(ZS_ENABLE_JIT)

  message("project name: ${project_name}")
  message("package version: ${project_version}")
  message("targets to install: ${INSTALL_TARGETS}")
  message("runtime dst: ${CMAKE_INSTALL_BINDIR}/${project_name}")
  message("archive dst: ${CMAKE_INSTALL_LIBDIR}/${project_name}")
  message("include dst: ${CMAKE_INSTALL_INCLUDEDIR}")
  message("install_rpath: $ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

  # libs, includes
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  set(ZENSIM_CONFIG_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/${project_name}-config-version.cmake")
  set(ZENSIM_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/${project_name}-config.cmake")
  write_basic_package_version_file(
    ${ZENSIM_CONFIG_VERSION_FILE}
    VERSION ${project_version}
    COMPATIBILITY AnyNewerVersion
  )
  set(cmakeModulesDir cmake)
  configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/CMake/${project_name}-config.cmake.in ${ZENSIM_CONFIG_FILE}
    INSTALL_DESTINATION share/${project_name}/cmake
    PATH_VARS cmakeModulesDir
    NO_SET_AND_CHECK_MACRO

    # INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${project_name}
  )
  install(TARGETS ${INSTALL_TARGETS}
    EXPORT ${project_name}-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${project_name}
    COMPONENT ${project_name}_Runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${project_name}
    COMPONENT ${project_name}_Runtime
    NAMELINK_COMPONENT ${project_name}_Development

    # OBJECTS DESTINATION ${CMAKE_INSTALL_LIBDIR}/${project_name}
    # COMPONENT ${project_name}_Runtime
    # NAMELINK_COMPONENT ${project_name}_Development
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${project_name}
    COMPONENT ${project_name}_Development
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}

    # PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/zensim # this relates to maxos framework
  )
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/zensim/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/zensim
    PATTERN "*.h|*.hpp|*.tpp|*.cuh|*.inl"
    PATTERN "zpc_tpls/*"

    # FILES_MATCHING PATTERN "*.h|*.hpp|*.tpp|*.cuh"
  )

  # config, version
  install(FILES ${ZENSIM_CONFIG_FILE} ${ZENSIM_CONFIG_VERSION_FILE}
    DESTINATION share/${project_name}/cmake

    # DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/${project_name}/cmake
  )

  # target file for build tree
  export(EXPORT ${project_name}-targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/cmake/${project_name}-targets.cmake
    NAMESPACE ${project_name}::
  )

  # target files export
  install(EXPORT ${project_name}-targets
    DESTINATION share/${project_name}/cmake
    NAMESPACE ${project_name}::

    # DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/${project_name}/cmake
  )

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/CMake/Uninstall.cmake
  )
endif(ZS_ENABLE_INSTALL)

# ---- Pacakge ----
# =================
if(ZS_ENABLE_PACKAGE)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # https://www.scivision.dev/cmake-cpack-basic/
    if(WIN32)
      set(_fmt TGZ ZIP)
    elseif(APPLE)
      set(_fmt TGZ)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      set(_fmt TGZ)
    endif()

    set(CPACK_GENERATOR ${_fmt})
    set(CPACK_SOURCE_GENERATOR ${_fmt})

    #
    # package info
    #
    set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
    set(CPACK_PACKAGE_NAME ${project_name})
    set(CPACK_PACKAGE_VENDOR "Zenus Co. Ltd.")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${project_description})
    set(CPACK_PACKAGE_VERSION_MAJOR ${project_version_major})
    set(CPACK_PACKAGE_VERSION_MINOR ${project_version_minor})
    set(CPACK_PACKAGE_VERSION_PATCH ${project_version_patch})
    set(CPACK_VERBATIM_VARIABLES TRUE)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})
    set(CPACK_PACKAGE_CONTACT "wangxl@zensim.cn/wxlwxl1993@zju.edu.cn")

    #
    # resource
    #
    # CPACK_RESOURCE_FILE_WELCOME
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
    set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

    set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/package")
    set(CPACK_PACKAGE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    string(TOLOWER ${CMAKE_SYSTEM_NAME} _sys)
    string(TOLOWER ${project_name} _project_lower)
    set(CPACK_PACKAGE_FILE_NAME "${_project_lower}-${_sys}")
    set(CPACK_SOURCE_PACKAGE_FILE_NAME "${_project_lower}-${project_version}")

    # not .gitignore as its regex syntax is distinct
    if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/.cpack_ignore)
      file(READ ${CMAKE_CURRENT_LIST_DIR}/.cpack_ignore _cpack_ignore)
      string(REGEX REPLACE "\n" ";" _cpack_ignore ${_cpack_ignore})
      set(CPACK_SOURCE_IGNORE_FILES "${_cpack_ignore}")
    endif()

    install(FILES ${CPACK_RESOURCE_FILE_README} ${CPACK_RESOURCE_FILE_LICENSE}
      DESTINATION share/docs/${project_name})

    include(CPack)

    #[=[
include(CPackIFW)

cpack_add_component(core GROUP GroupCore)
cpack_add_component(zpctool GROUP GroupTool)
#cpack_add_component(zpctool GROUP simulation)
#cpack_add_component( GROUP examples)
cpack_add_component_group(GroupCore) # for just 
cpack_add_component_group(GroupTool)
cpack_add_component_group(GroupSimulation)

cpack_ifw_configure_component(
  DISPLAY_NAME  en  Core
  DEPENDS com.examplecompany.product.Core.core
)
]=]
  endif()
endif(ZS_ENABLE_PACKAGE)
