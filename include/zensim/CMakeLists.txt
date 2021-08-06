set(ZENSIM_LIBRARY_CORE_SOURCE_FILES
    math/probability/Probability.cpp
    math/matrix/MatrixTransform.cpp
    math/matrix/MatrixCtor.cpp
    math/matrix/Matrix.cpp
    memory/Allocator.cpp
    resource/Resource.cpp
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
    ${ZENSIM_LIBRARY_CORE_SOURCE_FILES} ${ZENSIM_LIBRARY_CORE_INCLUDE_FILES}
)
if (ENABLE_PCH)
  target_precompile_headers(zpc INTERFACE 
    ${ZENSIM_LIBRARY_CORE_INCLUDE_FILES}
)
endif()
target_link_libraries(zpc PUBLIC zpc_cxx_deps)
target_compile_options(zpc # MSVC, GNU, Clang, Intel
  PUBLIC        $<$<COMPILE_LANGUAGE:CXX>: $<IF:$<CXX_COMPILER_ID:MSVC>, /O2, -O3 $<IF:$<CXX_COMPILER_ID:Clang>, , > >># -fuse-ld=lld -fvisibility=hidden># -flto=thin -fsanitize=cfi 
)
target_link_options(zpc
  PUBLIC        $<HOST_LINK:$<IF:$<CXX_COMPILER_ID:MSVC>, /O2 /MACHINE:X64 /OPT:REF /OPT:ICF, -O3 -m64 $<IF:$<CXX_COMPILER_ID:Clang>, , > >>
)
set_target_properties(zpc
  PROPERTIES  # LINKER_LANGUAGE CXX
              POSITION_INDEPENDENT_CODE ON
              # CXX_VISIBILITY_PRESET default
)
# set_property(TARGET zpc APPEND PROPERTY PUBLIC_HEADER "${ZENSIM_LIBRARY_CORE_INCLUDE_FILES} ${ZENSIM_LIBRARY_OMP_INCLUDE_FILES}")

###############
# zpc: openmp #
###############
if (ZS_ENABLE_OPENMP)

target_sources(zpc 
  PRIVATE   ${ZENSIM_LIBRARY_OMP_SOURCE_FILES} ${ZENSIM_LIBRARY_OMP_INCLUDE_FILES}
)
if (ENABLE_PCH)
  target_precompile_headers(zpc INTERFACE ${ZENSIM_LIBRARY_OMP_INCLUDE_FILES})
endif()

target_compile_definitions(zpc PUBLIC ZS_ENABLE_OPENMP=1)
target_compile_options(zpc # MSVC, GNU, Clang, Intel
  PUBLIC        $<$<COMPILE_LANGUAGE:CXX>: $<IF:$<CXX_COMPILER_ID:MSVC>, /openmp:llvm /openmp:experimental, -fopenmp $<IF:$<CXX_COMPILER_ID:Clang>, , > >>
)
# set_property(TARGET zpc APPEND PROPERTY PUBLIC_HEADER "${ZENSIM_LIBRARY_OMP_INCLUDE_FILES}")

endif(ZS_ENABLE_OPENMP)

#############
# zpc: cuda #
#############
if (ZS_ENABLE_CUDA)

target_sources(zpc 
  PRIVATE   ${ZENSIM_LIBRARY_CUDA_SOURCE_FILES} 
#  INTERFACE ${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES}
)
if (ENABLE_PCH)
  target_precompile_headers(zpc INTERFACE ${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES})
endif()

target_compile_definitions(zpc PUBLIC ZS_ENABLE_CUDA=1)
set_source_files_properties(${ZENSIM_LIBRARY_CUDA_SOURCE_FILES} ${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES} PROPERTIES  LANGUAGE CUDA)
target_link_libraries(zpc PUBLIC zpc_cuda_deps) # cuda_runtime cuda cudadevrt nvrtc cusolver cublas cusparse
target_compile_options(zpc 
  PRIVATE       $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
  PUBLIC        $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda --expt-relaxed-constexpr --default-stream=per-thread>
)
target_link_options(zpc 
  # PRIVATE       $<$<LINK_LANGUAGE:CUDA>:-dlto>
  INTERFACE       $<DEVICE_LINK:-dlto>
)
set_target_properties(zpc
  PROPERTIES  # LINKER_LANGUAGE CUDA
              CUDA_EXTENSIONS ON
              CUDA_SEPARABLE_COMPILATION ON
              CUDA_RESOLVE_DEVICE_SYMBOLS ON # https://devblogs.nvidia.com/building-cuda-applications-cmake/
              CUDA_ARCHITECTURES OFF
              #CUDA_VISIBILITY_PRESET default
)
# set_property(TARGET zpc APPEND PROPERTY PUBLIC_HEADER "${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES}")
target_compile_definitions(zpc
  PUBLIC    CMAKE_GENERATOR_PLATFORM=x64
)

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

target_link_libraries(zpctool PUBLIC zpc)
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

add_library(zs_cu_wrangler OBJECT)

target_sources(zs_cu_wrangler
  PRIVATE ${ZENSIM_LIBRARY_CUDA_WRANGLER_SOURCE_FILES}
)
target_compile_options(zs_cu_wrangler 
  PRIVATE       $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} -c --use_fast_math -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
)
set_source_files_properties(${ZENSIM_LIBRARY_CUDA_WRANGLER_INCLUDE_FILES} ${ZENSIM_LIBRARY_CUDA_WRANGLER_SOURCE_FILES} PROPERTIES  LANGUAGE CUDA)

target_link_libraries(zs_cu_wrangler PUBLIC zpc)
set(ZS_PTX_SOURCE_DIR cuda/simulation/wrangler)
set(ZS_PTX_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/zs_cu_wrangler.dir/${ZS_PTX_SOURCE_DIR})
set_target_properties(zs_cu_wrangler
  PROPERTIES  # LINKER_LANGUAGE CUDA
              CUDA_EXTENSIONS ON
              CUDA_SEPARABLE_COMPILATION ON
              CUDA_RESOLVE_DEVICE_SYMBOLS ON # https://devblogs.nvidia.com/building-cuda-applications-cmake/
              CUDA_ARCHITECTURES OFF
              CUDA_PTX_COMPILATION ON # https://stackoverflow.com/questions/55993745/how-should-i-get-cmake-to-also-create-ptx-files-for-my-kernels
              #CUDA_VISIBILITY_PRESET default
)
target_compile_definitions(zpc PUBLIC ZS_PTX_INCLUDE_DIR="${ZS_PTX_INCLUDE_DIR}")

# $<TARGET_OBJECTS:zs_cu_wrangler>
target_link_libraries(zensim INTERFACE zs_cu_wrangler)  # zs_cu_wrangler -> zensim

endif (ZS_ENABLE_CUDA)

# for header installation
# list_transform_prepend(ZENSIM_INTERFACE_FILES include/zensim/)
# message("${ZENSIM_INTERFACE_FILES}")
# PUBLIC_HEADER not working as expected.
# set_target_properties(zensim PROPERTIES PUBLIC_HEADER "${ZENSIM_INTERFACE_FILES}")
