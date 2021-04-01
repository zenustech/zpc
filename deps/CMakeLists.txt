find_package(Git)
include(FetchContent)
include(ExternalProject)
include(DownloadProject)

if (NOT EXISTS ${PROJECT_SOURCE_DIR}/include/zensim/tpls)
  file(ARCHIVE_EXTRACT INPUT ${PROJECT_SOURCE_DIR}/deps/tpls.7z
    DESTINATION ${PROJECT_SOURCE_DIR}/include/zensim/
  )
endif()

# opengl
find_package(OpenGL)
if (TARGET OpenGL::GL)
    set_target_properties(OpenGL::GL PROPERTIES IMPORTED_GLOBAL TRUE)
    target_link_libraries(project_dependencies INTERFACE OpenGL::GL)
else()
    message(FATAL_ERROR "OpenGL not found!")
endif()

# openmp
find_package(OpenMP)
if (TARGET OpenMP::OpenMP_CXX)
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES IMPORTED_GLOBAL TRUE)
    target_link_libraries(project_dependencies INTERFACE OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "OpenMP not found!")
endif()

# fmt
target_compile_definitions(project_dependencies INTERFACE FMT_HEADER_ONLY)
target_include_directories(project_dependencies INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/zensim/tpls/fmt/>
    $<INSTALL_INTERFACE:include/zensim/tpls/fmt/>
)

# spdlog
target_include_directories(project_dependencies INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/zensim/tpls/spdlog/>
    $<INSTALL_INTERFACE:include/zensim/tpls/spdlog/>
)

# Catch2
target_include_directories(project_dependencies INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/zensim/tpls/catch2/>
    $<INSTALL_INTERFACE:include/zensim/tpls/catch2/>
)

# gcem
target_include_directories(project_dependencies INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/zensim/tpls/gcem/>
    $<INSTALL_INTERFACE:include/zensim/tpls/gcem/>
)

# function_ref
target_include_directories(project_dependencies INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/zensim/tpls/function_ref/>
    $<INSTALL_INTERFACE:include/zensim/tpls/function_ref/>
)

# glm
target_include_directories(project_dependencies INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/zensim/tpls/glm/>
    $<INSTALL_INTERFACE:include/zensim/tpls/glm/>
)

# jitify
target_include_directories(project_dependencies INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/zensim/tpls/jitify/>
    $<INSTALL_INTERFACE:include/zensim/tpls/jitify/>
)

# rapidjson
target_include_directories(project_dependencies INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/zensim/tpls/rapidjson>
    $<INSTALL_INTERFACE:include/zensim/tpls/rapidjson>
)

# cxxopts
target_include_directories(project_dependencies INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/zensim/tpls/cxxopts>
    $<INSTALL_INTERFACE:include/zensim/tpls/cxxopts>
)

# magic_enum
target_include_directories(project_dependencies INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/zensim/tpls/magic_enum>
    $<INSTALL_INTERFACE:include/zensim/tpls/magic_enum>
)

#[=[=
the above are all header-only third-party dependency libraries
#=]=]

if(UNIX)
  find_package(Threads REQUIRED)
  set_target_properties(Threads::Threads PROPERTIES IMPORTED_GLOBAL TRUE)
  target_link_libraries(project_dependencies INTERFACE Threads::Threads)
endif()

# partio
add_subdirectory(${PROJECT_SOURCE_DIR}/include/zensim/tpls/partio ${CMAKE_CURRENT_BINARY_DIR}/tpls/partio EXCLUDE_FROM_ALL)
add_dependencies(project_dependencies zspartio)
target_link_libraries(project_dependencies INTERFACE zspartio)

# libnuma
find_package(NUMA REQUIRED)

# umpire
if (ZS_ENABLE_CUDA)
    set(ENABLE_CUDA On CACHE BOOL "Enable CUDA for Umpire build" FORCE)
endif()
set(ENABLE_NUMA On CACHE BOOL "Enable NUMA for Umpire build" FORCE)
set(ENABLE_OPENMP On CACHE BOOL "Enable CUDA for Umpire build" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/include/zensim/tpls/umpire ${CMAKE_CURRENT_BINARY_DIR}/tpls/umpire EXCLUDE_FROM_ALL)
if (ZS_ENABLE_CUDA)
    set_target_properties(zsumpire PROPERTIES  CUDA_ARCHITECTURES OFF)
    target_compile_options(zsumpire
      PRIVATE     $<$<COMPILE_LANGUAGE:CUDA>: -fopenmp --allow-unsupported-compiler --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread --use_fast_math -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
    )
endif()
add_dependencies(project_dependencies zsumpire)
target_link_libraries(project_dependencies INTERFACE zsumpire)

# openvdb
find_package(OpenVDB QUIET)
if (TARGET OpenVDB::openvdb)
    set_target_properties(OpenVDB::openvdb PROPERTIES IMPORTED_GLOBAL TRUE)
    target_link_libraries(project_dependencies INTERFACE OpenVDB::openvdb)
else()
    set(OpenVDB_FOUND 1)
    target_link_libraries(project_dependencies INTERFACE Half IlmImf tbb openvdb)
endif()
set(OpenVDB_FOUND ${OpenVDB_FOUND} PARENT_SCOPE)
