cmake_minimum_required(VERSION 3.18)
project(zsuse CXX CUDA)
#list(APPEND CMAKE_MODULE_PATH /usr/local/lib/cmake/Zensim/)
list(APPEND CMAKE_MODULE_PATH /usr/local/lib/cmake/OpenVDB/)

set(ZS_ENABLE_CUDA 1 CACHE BOOL "Enable Cuda")
#find_package(Zensim REQUIRED)
find_package(Zensim COMPONENTS zensim REQUIRED)
add_executable(zsuse main.cu)

target_compile_options(zsuse PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:-fopenmp --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread --use_fast_math -lineinfo --ptxas-options=-allow-expensive-optimizations=true>)

set_target_properties(zsuse PROPERTIES
	CUDA_EXTENSIONS ON
	CUDA_SEPARABLE_COMPILATION ON
	LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	POSITION_INDEPENDENT_CODE ON
	CUDA_ARCHITECTURES OFF)

target_compile_features(zsuse PRIVATE cuda_std_17)

message("Zensim found: ${Zensim_FOUND}")
message("cuda found: ${zscuda_FOUND}")
message("io found: ${zsio_FOUND}")
message("vdb found: ${zsvdb_FOUND}")
message(${Zensim_INCLUDE_DIR})
get_property(INTERFACE_INCLUDES TARGET Zensim::zsproj_deps PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
message(${INTERFACE_INCLUDES})

#target_include_directories(zsuse PRIVATE ${INTERFACE_INCLUDES})
target_link_libraries(zsuse PRIVATE Zensim::zensim)

