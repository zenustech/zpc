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
  LANGUAGES CXX)

if(MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

list(APPEND CMAKE_MODULE_PATH /usr/local/lib/cmake/OpenVDB/) # openvdb's default linux install location
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_BINARY_DIR})

include(GNUInstallDirs)

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

option(ZS_ENABLE_PCH "Enable Precompiled Headers" OFF)
option(ZS_ENABLE_TEST "Enable Tests" OFF)
option(ZS_ENABLE_DOC "Enable Doxygen Documentation Generation" OFF)
option(ZS_ENABLE_OFB_ACCESS_CHECK "Enable out-of-bound access check" OFF)
option(ZS_BUILD_SHARED_LIBS "Enable compilation of shared libraries" OFF)
option(ZS_ENABLE_CUDA "Enable cuda backend" ON)
option(ZS_ENABLE_OPENMP "Enable openmp backend" ON)
option(ZS_ENABLE_OPENVDB "Enable openvdb-related build" ON)
option(ZS_ENABLE_SYCL_COMPUTECPP "Enable SYCL[ComputeCpp] backend" ON)
option(ZS_ENABLE_SYCL_DPCPP "Enable SYCL[Intel-DPC++] backend" OFF)
option(ZS_ENABLE_SYCL "Enable SYCL[Clang-DPC++] backend" OFF)
option(ZS_ENABLE_OPENCL "Enable OpenCL backend" OFF)
option(ZS_PROPAGATE_DEPS "Pass on dependencies" ON)
option(ZS_AUTO_DETECT_CUDA_ARCH "Let cmake automatically decides target cuda architecture" ON)

option(ZS_ENABLE_INSTALL "Install targets" Off)
option(ZS_ENABLE_PACKAGE "Build package" Off)

# option(ZENSIM_CORE_SHARED "Build dynamically linked version of the core library." ON)
# option(ZENSIM_CORE_STATIC "Build statically linked version of the core library." ON)

# project base targets
# ====================
include(CMake/Utility.cmake)

# include(CMake/ClangSupport.cmake)
include(CMake/Cache.cmake)
include(CMake/StaticAnalyzers.cmake)
include(CMake/Sanitizers.cmake)
include(CMake/CompilerWarnings.cmake)
include(CMake/Doxygen.cmake)
enable_doxygen()

# ---- dependencies ----
# ======================
add_subdirectory(include/zensim/zpc_tpls) # build zsproj_deps target
enable_sanitizers(zsproj_deps)

if(ENABLE_PCH)
  # This sets a global PCH parameter, each project will build its own PCH, which
  # is a good idea if any #define's change
  # consider breaking this out per project as necessary
  target_precompile_headers(zsproj_deps INTERFACE <vector> <string> <map> <utility> <cstdio>)
endif()

# ---- build setup -----
# ======================
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

add_library(zpc_cxx_deps INTERFACE)
target_compile_features(zpc_cxx_deps INTERFACE cxx_std_17)
target_link_libraries(zpc_cxx_deps INTERFACE zsproj_deps)

if(ZS_ENABLE_OFB_ACCESS_CHECK)
  target_compile_definitions(zpc_cxx_deps
    INTERFACE ZS_ENABLE_OFB_ACCESS_CHECK=1
  )
else(ZS_ENABLE_OFB_ACCESS_CHECK)
  target_compile_definitions(zpc_cxx_deps
    INTERFACE ZS_ENABLE_OFB_ACCESS_CHECK=0
  )
endif(ZS_ENABLE_OFB_ACCESS_CHECK)

target_compile_definitions(zpc_cxx_deps
  INTERFACE AssetDirPath="${PROJECT_SOURCE_DIR}/zpc_assets/"
)

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

# ---- Library ----
# =================
add_subdirectory(include/zensim)

set(ZS_ENABLE_OPENGL ${ZS_ENABLE_OPENGL} PARENT_SCOPE)
set(ZS_ENABLE_SYCL_COMPUTECPP ${ZS_ENABLE_SYCL_COMPUTECPP} PARENT_SCOPE)

if(ZS_ENABLE_SYCL_COMPUTECPP)
  set(COMPUTECPP_DEVICE_COMPILER_FLAGS ${COMPUTECPP_DEVICE_COMPILER_FLAGS} PARENT_SCOPE)
endif()

set(ZS_ENABLE_SYCL_DPCPP ${ZS_ENABLE_SYCL_DPCPP} PARENT_SCOPE)
set(ZS_ENABLE_SYCL ${ZS_ENABLE_SYCL} PARENT_SCOPE)
set(ZS_ENABLE_CUDA ${ZS_ENABLE_CUDA} PARENT_SCOPE)
set(ZS_ENABLE_OPENCL ${ZS_ENABLE_OPENCL} PARENT_SCOPE)
set(ZS_ENABLE_PTHREADS ${ZS_ENABLE_PTHREADS} PARENT_SCOPE)
set(ZS_ENABLE_OPENMP ${ZS_ENABLE_OPENMP} PARENT_SCOPE)
set(ZS_ENABLE_OPENVDB ${ZS_ENABLE_OPENVDB} PARENT_SCOPE)
set(ZS_ENABLE_PARTIO ${ZS_ENABLE_PARTIO} PARENT_SCOPE)
set(ZS_ENABLE_MSHIO ${ZS_ENABLE_MSHIO} PARENT_SCOPE)

message("cmake configure summary")
message("project name: ${project_name}")
message("package version: ${project_version}")
message("build options: pthreads: ${ZS_ENABLE_PTHREADS}, sycl: ${ZS_ENABLE_SYCL}, sycl(dpcpp): ${ZS_ENABLE_SYCL_DPCPP}, sycl(computecpp): ${ZS_ENABLE_SYCL_COMPUTECPP}, cuda: ${ZS_ENABLE_CUDA}(${CUDAToolkit_VERSION}), omp: ${ZS_ENABLE_OPENMP}, cl: ${ZS_ENABLE_OPENCL}, openvdb ${ZS_ENABLE_OPENVDB}, opengl ${ZS_ENABLE_OPENGL}, partio ${ZS_ENABLE_PARTIO}, mshio ${ZS_ENABLE_MSHIO}")

# ----  Docs   ----
# =================
if(ZS_ENABLE_DOC)
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
  set(ZS_INSTALL_DEP_TARGETS zspartio)

  if(ZS_ENABLE_OPENVDB)
    set(ZS_INSTALL_DEP_TARGETS ${ZS_INSTALL_DEP_TARGETS} zsopenvdb)
  endif(ZS_ENABLE_OPENVDB)

  set(INSTALL_TARGETS zensim zpc zpctool zpc_cxx_deps zsproj_deps ${ZS_INSTALL_DEP_TARGETS})

  if(ZS_ENABLE_OPENMP)
    set(INSTALL_TARGETS ${INSTALL_TARGETS} zpcomp zpc_omp_deps)
  endif(ZS_ENABLE_OPENMP)

  if(ZS_ENABLE_CUDA)
    set(INSTALL_TARGETS ${INSTALL_TARGETS} zpccuda zpc_cuda_deps)
  endif(ZS_ENABLE_CUDA)

  if(ZS_ENABLE_SYCL_COMPUTECPP OR ZS_ENABLE_SYCL_DPCPP OR ZS_ENABLE_SYCL)
    set(INSTALL_TARGETS ${INSTALL_TARGETS} zpcsycl zpc_sycl_deps)
  endif() # ZS_ENABLE_SYCL

  if(ZS_ENABLE_OPENCL)
    set(INSTALL_TARGETS ${INSTALL_TARGETS} zpccl zpc_cl_deps)
  endif(ZS_ENABLE_OPENCL)

  message("project name: ${project_name}")
  message("package version: ${project_version}")
  message("targets to install: ${INSTALL_TARGETS}")
  message("runtime dst: ${CMAKE_INSTALL_BINDIR}/${project_name}")
  message("archive dst: ${CMAKE_INSTALL_LIBDIR}/${project_name}")
  message("include dst: ${CMAKE_INSTALL_INCLUDEDIR}")
  message("install_rpath: $ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

  # libs, includes
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
#]=]
  endif()
endif(ZS_ENABLE_PACKAGE)