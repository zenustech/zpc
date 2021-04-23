cmake_minimum_required(VERSION 3.18)    # support relative path, cuda_std_17, archive manipulation

message("===============================================================")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

# project meta
include("project_info.in")
project(${project_name} 
    VERSION     ${project_version}
    DESCRIPTION ${project_description}
    LANGUAGES   CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake "/usr/local/lib/cmake/OpenVDB/")

# options
# ========
option(CMAKE_EXPORT_COMPILE_COMMANDS "" 1)

option(ENABLE_PCH "Enable Precompiled Headers" OFF)
option(BUILD_SHARED_LIBS "Enable compilation of shared libraries" OFF)
set(ZS_ENABLE_CUDA On CACHE BOOL "Enable cuda-related build")

option(ZENSIM_INSTALL "Install targets" On)
option(ZENSIM_CORE_SHARED "Build dynamically linked version of the core library." ON)
option(ZENSIM_CORE_STATIC "Build statically linked version of the core library." ON)

# project base targets
# ====================
add_library(zsproj_deps INTERFACE)
add_library(zsproj_cxx_deps INTERFACE)

include(CMake/ClangSupport.cmake)
include(CMake/Cache.cmake)
include(CMake/StaticAnalyzers.cmake)
include(CMake/Sanitizers.cmake)
enable_sanitizers(zsproj_deps)
include(CMake/CompilerWarnings.cmake)
# set_project_warnings(zsproj_deps)
include(CMake/Doxygen.cmake)
enable_doxygen()
if (ENABLE_PCH)
  # This sets a global PCH parameter, each project will build its own PCH, which
  # is a good idea if any #define's change
  # consider breaking this out per project as necessary 
  target_precompile_headers(zsproj_deps INTERFACE <vector> <string> <map> <utility> <cstdio>)
endif()

# ---- dependencies ---- 
# ======================
add_subdirectory(deps)

target_compile_features(zsproj_cxx_deps INTERFACE cxx_std_17)
target_link_libraries(zsproj_cxx_deps INTERFACE zsproj_deps)
target_compile_definitions(zsproj_cxx_deps
    INTERFACE   AssetDirPath="${PROJECT_SOURCE_DIR}/Data/"
)
if (ZS_ENABLE_CUDA)
  add_library(zsproj_cuda_deps INTERFACE)
  target_compile_features(zsproj_cuda_deps INTERFACE cuda_std_17)
  target_link_libraries(zsproj_cuda_deps INTERFACE zsproj_deps)
endif()

# ---- binaries ----
# ==================
include(setup_cpp.cmake)
if (ZS_ENABLE_CUDA)
    include(setup_cuda.cmake)
endif()

# ---- Library ----
# =================
target_include_directories(zsproj_deps INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
)
add_subdirectory(include)

# ---- Install ----
# =================
if (ZENSIM_INSTALL)
include(GNUInstallDirs)

#set(ZENSIM_CMAKE_MODULES CMake/FindZensim.cmake)
#install(FILES ${ZENSIM_CMAKE_MODULES} DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${project_name})

set(ZSUMPIRE_INSTALL_TARGETS zsumpire zscamp zsumpire_alloc zsumpire_resource zsumpire_strategy zsumpire_op zsumpire_util zsumpire_tpl_judy)
set(ZSPARTIO_INSTALL_TARGETS zspartio)

set(INSTALL_TARGETS zensim zscore zstool zsproj_cxx_deps zsproj_deps ${ZSUMPIRE_INSTALL_TARGETS} ${ZSPARTIO_INSTALL_TARGETS})
if (ZS_ENABLE_CUDA)
  set(INSTALL_TARGETS ${INSTALL_TARGETS} zscuda zsproj_cuda_deps cuda_runtime cuda)
endif()

message("project name: ${project_name}")
message("package version: ${project_version}")
message("targets to install: ${INSTALL_TARGETS}")
message("runtime dst: ${CMAKE_INSTALL_BINDIR}/${project_name}")
message("archive dst: ${CMAKE_INSTALL_LIBDIR}/${project_name}")
message("include dst: ${CMAKE_INSTALL_INCLUDEDIR}")

if (NOT APPLE)
  set(CMAKE_INSTALL_RPATH $ORIGIN)
endif()

# libs, includes
install(TARGETS ${INSTALL_TARGETS}
  EXPORT ${project_name}-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${project_name}
          COMPONENT ${project_name}_Runtime
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${project_name}
          COMPONENT ${project_name}_Runtime
          NAMELINK_COMPONENT ${project_name}_Development
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${project_name}
          COMPONENT ${project_name}_Development
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/zensim/ 
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/zensim)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/zensim/tpls/ 
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/zensim/tpls)

# config, version
include(CMakePackageConfigHelpers)
set(ZENSIM_CONFIG_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/${project_name}ConfigVersion.cmake")
set(ZENSIM_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/${project_name}Config.cmake")
write_basic_package_version_file(
  ${ZENSIM_CONFIG_VERSION_FILE}
  VERSION ${project_version}
  COMPATIBILITY AnyNewerVersion
)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/CMake/ZensimConfig.cmake.in
  ${ZENSIM_CONFIG_FILE}
  #INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${project_name}
  INSTALL_DESTINATION share/${project_name}/cmake}
)
install(
  FILES ${ZENSIM_CONFIG_FILE} ${ZENSIM_CONFIG_VERSION_FILE} 
  DESTINATION share/${project_name}/cmake
  # DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/${project_name}/cmake
)
# target files export
install(EXPORT ${project_name}-targets
  FILE ${project_name}Targets.cmake
  NAMESPACE ${project_name}:: 
  DESTINATION share/${project_name}/cmake
  # DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/${project_name}/cmake
)

# target file for build tree
export(EXPORT ${project_name}-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/cmake/${project_name}Targets.cmake
  NAMESPACE ${project_name}:: 
)
endif()