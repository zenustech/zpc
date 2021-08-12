set(ZENSIM_LIBRARY_CORE_SOURCE_FILES
    math/probability/Probability.cpp
    math/matrix/MatrixTransform.cpp
    math/matrix/MatrixCtor.cpp
    math/matrix/Matrix.cpp
    memory/Allocator.cpp
    profile/CppTimers.cpp
    execution/Stacktrace.cpp
    execution/ExecutionPolicy.cpp
    execution/ConcurrencyPrimitive.cpp
    # simulation
    simulation/mpm/Simulator.cpp
    simulation/particle/Query.cpp
    simulation/sparsity/SparsityCompute.cpp
    memory/MemOps.cpp
)
set(ZENSIM_LIBRARY_IO_SOURCE_FILES
    io/IO.cpp
    # simulation
    simulation/init/Scene.cpp
)
set(ZENSIM_LIBRARY_TOOL_SOURCE_FILES
    geometry/VdbLevelSet_Point.cpp
    geometry/VdbLevelSet_LoadPhi.cpp
    geometry/VdbLevelSet_LoadPhiVel.cpp
    geometry/VdbSampler_SampleFromLevelSet.cpp
    geometry/VdbSampler_MeshToLevelSet.cpp
    geometry/VdbSampler_FileToLevelSet.cpp
)
set(ZENSIM_LIBRARY_CUDA_SOURCE_FILES
    cuda/Cuda.cu
	  cuda/Allocators.cu
	  cuda/profile/CudaTimers.cu
    cuda/math/matrix/Matrix.cpp
    cuda/simulation/particle/Query.cpp
    cuda/simulation/sparsity/SparsityCompute.cpp
    cuda/memory/MemOps.cpp
    cuda/simulation/wrangler/Wrangler.cpp
)
set(ZENSIM_LIBRARY_OMP_SOURCE_FILES
    omp/math/matrix/MatrixCtor.cpp
    omp/math/matrix/MatrixTransform.cpp
    omp/simulation/particle/Query.cpp
    omp/simulation/sparsity/SparsityCompute.cpp
    omp/execution/ExecutionPolicy.cpp
)
set(ZENSIM_LIBRARY_ZPC_SOURCE_FILES
    resource/Resource.cpp
)

## headers
set(ZENSIM_LIBRARY_CORE_INCLUDE_FILES
    # container
    container/DenseGrid.hpp
    container/RingBuffer.hpp
    container/TileVector.hpp
    container/HashTable.hpp
    container/Vector.hpp
    container/Bvh.hpp
    container/Bvtt.hpp
    container/IndexBuckets.hpp
    geometry/Structure.hpp
    geometry/Structurefree.hpp
    # execution
    execution/Concurrency.h
    execution/ExecutionPolicy.hpp
    execution/Stacktrace.hpp
    execution/Atomics.hpp
    execution/Intrinsics.hpp
    # geometry
    geometry/AnalyticLevelSet.h
    geometry/BoundingVolumeInterface.hpp
    geometry/Collider.h
    geometry/GenericLevelSet.h
    geometry/LevelSet.h
    geometry/LevelSetInterface.h
    geometry/LevelSetSequence.hpp
    geometry/Mesh.hpp
    geometry/PoissonDisk.hpp
    geometry/SparseLevelSet.hpp
    # math
    math/bit/Bits.h
    math/linear/ConjugateGradient.hpp
    math/linear/LinearOperators.hpp
    math/matrix/MatrixUtils.h
    math/matrix/Utility.h
    math/matrix/MatrixTransform.hpp
    math/matrix/MatrixCtor.hpp
    math/matrix/Matrix.hpp
    math/matrix/Givens.hpp
    math/matrix/qr.hpp
    math/probability/Probability.h
    math/Hash.hpp
    math/MathUtils.h
    math/RandomNumber.hpp
    math/Rotation.hpp
    math/Vec.h
    # memory
    memory/MemOps.hpp
    memory/Allocator.h
    memory/MemoryResource.h
    # meta
    meta/ControlFlow.h
    meta/Functional.h
    meta/Meta.h
    meta/Relationship.h
    meta/Sequence.h
    # profile
    profile/CppTimers.hpp
    # resource
    resource/Resource.h
    # types
    types/Function.h
    types/Iterator.h
    types/Optional.h
    types/Polymorphism.h
    types/Property.h
    types/RuntimeStructurals.hpp
    types/Tuple.h
    types/BuilderBase.hpp
    types/SmallVector.hpp
    types/SourceLocation.hpp
    types/View.h
    Logger.hpp
    Platform.hpp
    Reflection.h
    Singleton.h
    TypeAlias.hpp
    # simulation
    physics/ConstitutiveModel.hpp
    simulation/mpm/Simulator.hpp
    simulation/transfer/P2G.hpp
    simulation/transfer/G2P.hpp
    simulation/grid/GridOp.hpp
    simulation/particle/ParticleOp.hpp
    simulation/particle/Query.hpp
    simulation/sparsity/SparsityOp.hpp
    simulation/sparsity/SparsityCompute.hpp
    simulation/Utils.hpp
    # tpp
    simulation/particle/Query.tpp
    simulation/sparsity/SparsityCompute.tpp
    container/BvhImpl.tpp
)
set(ZENSIM_LIBRARY_IO_INCLUDE_FILES
    io/IO.h
    io/MeshIO.hpp
    io/ParticleIO.hpp
    # simulation
    simulation/init/Scene.hpp
)
set(ZENSIM_LIBRARY_TOOL_INCLUDE_FILES
    geometry/VdbLevelSet.h
    geometry/VdbSampler.h
)
set(ZENSIM_LIBRARY_CUDA_INCLUDE_FILES
    cuda/Cuda.h
    cuda/DeviceUtils.cuh
    cuda/HostUtils.hpp
    cuda/CudaLaunchConfig.cuh
    cuda/Allocators.cuh
    cuda/profile/CudaTimers.cuh
    cuda/memory/MemOps.hpp
    cuda/math/matrix/EigenDecomposition.cuh
    cuda/math/matrix/svd.cuh
    cuda/math/matrix/Matrix.hpp # recent
    cuda/execution/ExecutionPolicy.cuh
    # container
    cuda/geometry/Structurefree.hpp # recent
    cuda/algorithm/MappingKernels.cuh
    cuda/algorithm/SortKernels.cuh
    # simulation
    cuda/simulation/transfer/P2G.hpp
    cuda/simulation/particle/Query.hpp
    # physics
    cuda/physics/ConstitutiveModel.hpp
    # wrangler
    cuda/simulation/wrangler/Wrangler.hpp
)
set(ZENSIM_LIBRARY_OMP_INCLUDE_FILES
    omp/execution/ExecutionPolicy.hpp
    omp/math/matrix/MatrixCtor.hpp
    omp/math/matrix/MatrixTransform.hpp
)
# wrangler
set(ZENSIM_LIBRARY_CUDA_WRANGLER_INCLUDE_FILES
)
set(ZENSIM_LIBRARY_CUDA_WRANGLER_SOURCE_FILES
    cuda/simulation/wrangler/VolumeWrangler.cpp
    cuda/simulation/wrangler/ParticleWrangler.cpp
)

#set(CMAKE_CXX_VISIBILITY_PRESET     hidden)
#set(CMAKE_VISIBILITY_INLINES_HIDDEN NO)
#include(GenerateExportHeader)
#generate_export_header(zensim EXPORT_FILE_NAME ZensimExport.hpp)
#message("building shared?(${BUILD_SHARED_LIBS}) library")

add_library(zensim INTERFACE)

#############
# zpc: core #
#############
if (ZS_BUILD_SHARED_LIBS)
  add_library(zpccore SHARED)
else ()
  add_library(zpccore STATIC)
endif ()

target_sources(zpccore
  PRIVATE 
    ${ZENSIM_LIBRARY_CORE_SOURCE_FILES} ${ZENSIM_LIBRARY_CORE_INCLUDE_FILES}
)
if (ENABLE_PCH)
  target_precompile_headers(zpccore INTERFACE 
    ${ZENSIM_LIBRARY_CORE_INCLUDE_FILES}
  )
endif()
target_link_libraries(zpccore PUBLIC zpc_cxx_deps)
target_compile_options(zpccore
  PUBLIC        $<$<COMPILE_LANGUAGE:CXX>: $<IF:$<CXX_COMPILER_ID:MSVC>, /O2, -O3 $<IF:$<CXX_COMPILER_ID:Clang>, , > >># -fuse-ld=lld -fvisibility=hidden># -flto=thin -fsanitize=cfi 
)
target_link_options(zpccore
  PUBLIC        $<HOST_LINK:$<IF:$<CXX_COMPILER_ID:MSVC>, /MACHINE:X64 /OPT:REF /OPT:ICF, -O3 -m64 $<IF:$<CXX_COMPILER_ID:Clang>, , > >>
)
set_target_properties(zpccore
  PROPERTIES  POSITION_INDEPENDENT_CODE ON
              # CXX_VISIBILITY_PRESET default
)

if (ZS_ENABLE_OPENMP)
  target_compile_definitions(zpccore PUBLIC ZS_ENABLE_OPENMP=1)
else()
  target_compile_definitions(zpccore PUBLIC ZS_ENABLE_OPENMP=0)
endif(ZS_ENABLE_OPENMP)

if (ZS_ENABLE_CUDA)
  target_compile_definitions(zpccore PUBLIC ZS_ENABLE_CUDA=1)
else()
  target_compile_definitions(zpccore PUBLIC ZS_ENABLE_CUDA=0)
endif(ZS_ENABLE_CUDA)

###############
# zpc: openmp #
###############
if (ZS_ENABLE_OPENMP)

if (ZS_BUILD_SHARED_LIBS)
  add_library(zpcomp SHARED)
else ()
  add_library(zpcomp STATIC)
endif ()
target_sources(zpcomp 
  PRIVATE   ${ZENSIM_LIBRARY_OMP_SOURCE_FILES} ${ZENSIM_LIBRARY_OMP_INCLUDE_FILES}
)
if (ENABLE_PCH)
  target_precompile_headers(zpcomp INTERFACE ${ZENSIM_LIBRARY_OMP_INCLUDE_FILES})
endif()
target_link_libraries(zpcomp PUBLIC zpccore zpc_omp_deps)
target_compile_options(zpcomp # MSVC, GNU, Clang, Intel
  PRIVATE   $<$<COMPILE_LANGUAGE:CXX>: $<IF:$<CXX_COMPILER_ID:MSVC>, /openmp:llvm /openmp:experimental, -fopenmp $<IF:$<CXX_COMPILER_ID:Clang>, , > >>
)

endif(ZS_ENABLE_OPENMP)

#############
# zpc: cuda #
#############
if (ZS_ENABLE_CUDA)

if (ZS_BUILD_SHARED_LIBS)
  add_library(zpccuda SHARED)
else ()
  add_library(zpccuda STATIC)
endif ()
target_sources(zpccuda 
  PRIVATE   ${ZENSIM_LIBRARY_CUDA_SOURCE_FILES} #  INTERFACE ${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES}
)
if (ENABLE_PCH)
  target_precompile_headers(zpccuda INTERFACE ${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES})
endif()
set_source_files_properties(${ZENSIM_LIBRARY_CUDA_SOURCE_FILES} ${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES} PROPERTIES  LANGUAGE CUDA)
target_link_libraries(zpccuda PUBLIC zpccore zpc_cuda_deps) # cuda_runtime cuda cudadevrt nvrtc cusolver cublas cusparse
target_compile_options(zpccuda 
  PRIVATE       $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
# https://stackoverflow.com/questions/64523302/cuda-missing-return-statement-at-end-of-non-void-function-in-constexpr-if-fun
  PUBLIC        $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda --expt-relaxed-constexpr --default-stream=per-thread -Xcudafe "--diag_suppress=implicit_return_from_non_void_function">
)
target_link_options(zpccuda 
  PRIVATE       $<DEVICE_LINK:-dlto>
)
set_target_properties(zpccuda
  PROPERTIES  # LINKER_LANGUAGE CUDA
              CUDA_EXTENSIONS ON
              CUDA_SEPARABLE_COMPILATION ON
              CUDA_RESOLVE_DEVICE_SYMBOLS ON # https://devblogs.nvidia.com/building-cuda-applications-cmake/
              CUDA_ARCHITECTURES OFF
              #CUDA_VISIBILITY_PRESET default
)
target_compile_definitions(zpccuda
  PUBLIC    CMAKE_GENERATOR_PLATFORM=x64
)

endif(ZS_ENABLE_CUDA)

#######
# zpc #
#######
if (ZS_BUILD_SHARED_LIBS)
  add_library(zpc SHARED)
else ()
  add_library(zpc STATIC)
endif ()

target_sources(zpc
  PRIVATE 
    ${ZENSIM_LIBRARY_ZPC_SOURCE_FILES}
)
target_link_libraries(zpc PUBLIC zpccore)
if (ZS_ENABLE_OPENMP)
  target_link_libraries(zpc PUBLIC zpcomp)
endif(ZS_ENABLE_OPENMP)
if (ZS_ENABLE_CUDA)
  target_link_libraries(zpc PUBLIC zpccuda)
endif(ZS_ENABLE_CUDA)

target_link_libraries(zensim INTERFACE zpc)  # zpc -> zensim

# openvdb, partio
###########
# zpctool #
###########
if (ZS_BUILD_SHARED_LIBS)
  add_library(zpctool SHARED)
else ()
  add_library(zpctool STATIC)
endif ()

target_sources(zpctool
  PRIVATE 
    ${ZENSIM_LIBRARY_IO_SOURCE_FILES}
    ${ZENSIM_LIBRARY_IO_INCLUDE_FILES}
)
if (ENABLE_PCH)
  target_precompile_headers(zpctool INTERFACE ${ZENSIM_LIBRARY_TOOL_INCLUDE_FILES} ${ZENSIM_LIBRARY_IO_INCLUDE_FILES})
endif(ENABLE_PCH)

target_link_libraries(zpctool PUBLIC zpc zspartio)
set_target_properties(zpctool
  PROPERTIES  POSITION_INDEPENDENT_CODE ON
)
# set_property(TARGET zpctool APPEND PROPERTY PUBLIC_HEADER "${ZENSIM_LIBRARY_TOOL_INCLUDE_FILES} ${ZENSIM_LIBRARY_IO_INCLUDE_FILES}")

if (ZS_ENABLE_OPENVDB)

  target_link_libraries(zpctool PRIVATE zsopenvdb)
  target_sources(zpctool
    PRIVATE 
      ${ZENSIM_LIBRARY_TOOL_SOURCE_FILES} 
      ${ZENSIM_LIBRARY_TOOL_INCLUDE_FILES} 
  )

endif(ZS_ENABLE_OPENVDB)

target_link_libraries(zensim INTERFACE zpctool)  # zpctool -> zensim

##################
# zs_cu_wrangler #
##################

if (ZS_ENABLE_CUDA)

if (MSVC)
target_compile_definitions(zpccuda PUBLIC ZS_PTX_INCLUDE_DIR="")

else()

add_library(zs_cu_wrangler OBJECT)

target_sources(zs_cu_wrangler
  PRIVATE ${ZENSIM_LIBRARY_CUDA_WRANGLER_SOURCE_FILES}
)
target_compile_options(zs_cu_wrangler 
  PRIVATE       $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} -ptx --use_fast_math -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
)
set_source_files_properties(${ZENSIM_LIBRARY_CUDA_WRANGLER_INCLUDE_FILES} ${ZENSIM_LIBRARY_CUDA_WRANGLER_SOURCE_FILES} PROPERTIES  LANGUAGE CUDA)

target_link_libraries(zs_cu_wrangler PRIVATE zpccuda)
set(ZS_PTX_SOURCE_DIR cuda/simulation/wrangler)
set(ZS_PTX_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/zs_cu_wrangler.dir/${ZS_PTX_SOURCE_DIR})
set_target_properties(zs_cu_wrangler
  PROPERTIES  LINKER_LANGUAGE CUDA
              CUDA_EXTENSIONS ON
              CUDA_SEPARABLE_COMPILATION ON
              CUDA_RESOLVE_DEVICE_SYMBOLS ON # https://devblogs.nvidia.com/building-cuda-applications-cmake/
              CUDA_ARCHITECTURES OFF
              CUDA_PTX_COMPILATION ON # https://stackoverflow.com/questions/55993745/how-should-i-get-cmake-to-also-create-ptx-files-for-my-kernels
              #CUDA_VISIBILITY_PRESET default
)
target_compile_definitions(zpccuda PUBLIC ZS_PTX_INCLUDE_DIR="${ZS_PTX_INCLUDE_DIR}")

# $<TARGET_OBJECTS:zs_cu_wrangler>
target_link_libraries(zensim INTERFACE zs_cu_wrangler)  # zs_cu_wrangler -> zensim
endif(MSVC)

endif (ZS_ENABLE_CUDA)

# for header installation
# list_transform_prepend(ZENSIM_INTERFACE_FILES include/zensim/)
# message("${ZENSIM_INTERFACE_FILES}")
# PUBLIC_HEADER not working as expected.
# set_target_properties(zensim PROPERTIES PUBLIC_HEADER "${ZENSIM_INTERFACE_FILES}")
