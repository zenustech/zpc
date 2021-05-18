set(ZENSIM_LIBRARY_CORE_SOURCE_FILES
    DynamicLoader.cpp
    math/probability/Probability.cpp
    math/matrix/MatrixTransform.cpp
    math/matrix/MatrixCtor.cpp
    math/matrix/Matrix.cpp
    memory/Allocator.cpp
    profile/CppTimers.cpp
    execution/Stacktrace.cpp
    execution/ExecutionPolicy.cpp
    resource/Resource.cpp
    # simulation
    simulation/mpm/Simulator.cpp
    simulation/sparsity/SparsityCompute.cpp
    memory/operations/Copy.cpp
)
set(ZENSIM_LIBRARY_IO_SOURCE_FILES
    io/IO.cpp
    io/stb_image.cpp
    # simulation
    simulation/init/Boundary.cpp
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
	cuda/DeviceUtils.cu
    cuda/AuxiliaryLibs.cu
	cuda/profile/CudaTimers.cu
    cuda/math/matrix/Matrix.cpp
    cuda/math/matrix/MatrixCtor.cpp
    cuda/math/matrix/MatrixTransform.cpp
    cuda/simulation/mpm/Simulator.cpp
    cuda/simulation/sparsity/SparsityCompute.cpp
    cuda/memory/operations/Copy.cpp
)
set(ZENSIM_LIBRARY_OMP_SOURCE_FILES
    omp/math/matrix/MatrixCtor.cpp
    omp/math/matrix/MatrixTransform.cpp
)

## headers
set(ZENSIM_LIBRARY_CORE_INCLUDE_FILES
    # container
    container/fsm/FiniteStateMachine.h
    container/Container.hpp
    container/DenseGrid.hpp
    container/RingBuffer.hpp
    container/TileVector.hpp
    container/HashTable.hpp
    container/Vector.hpp
    geometry/Structure.hpp
    geometry/Structurefree.hpp
    # execution
    execution/Concurrency.h
    execution/ExecutionPolicy.hpp
    execution/Stacktrace.hpp
    execution/SyncPrimitive.hpp
    # geometry
    geometry/AnalyticLevelSet.h
    geometry/Collider.h
    geometry/GenericLevelSet.h
    geometry/GeometrySampler.h
    geometry/LevelSet.h
    geometry/LevelSetInterface.h
    geometry/LevelSetSequence.hpp
    geometry/Mesh.hpp
    geometry/PoissonDisk.hpp
    geometry/SparseLevelSet.hpp
    geometry/AdaptiveLevelSet.hpp
    # math
    math/bit/Bits.h
    math/matrix/MatrixUtils.h
    math/matrix/Utility.h
    math/matrix/MatrixTransform.hpp
    math/matrix/MatrixCtor.hpp
    math/matrix/Matrix.hpp
    math/probability/Probability.h
    math/Hash.hpp
    math/MathUtils.h
    math/Matrix.hpp
    math/RandomNumber.hpp
    math/Rotation.hpp
    math/ValueSequence.hpp
    math/Vec.h
    # memory
    # memory/Memory.hpp
    memory/operations/Copy.hpp
    memory/Allocator.h
    memory/MemObj.h
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
    types/Event.hpp
    types/Function.h
    types/Inherit.h
    types/Iterator.h
    types/Object.h
    types/Optional.h
    types/Polymorphism.h
    types/Property.h
    types/RuntimeStructurals.hpp
    types/Structural.h
    types/StructuralAuxiliary.h
    types/StructuralDeclaration.h
    types/Tuple.h
    types/Value.h
    types/BuilderBase.hpp
    types/SmallVector.hpp
    DynamicLoader.h
    Logger.hpp
    Platform.hpp
    Reflection.h
    Singleton.h
    TypeAlias.hpp
    zensim.hpp
    # simulation
    simulation/mpm/Simulator.hpp
    simulation/physics/ConstitutiveModel.hpp
    simulation/transfer/P2G.hpp
    simulation/transfer/G2P.hpp
    simulation/grid/GridOp.hpp
    simulation/particle/ParticleOp.hpp
    simulation/sparsity/SparsityOp.hpp
    simulation/sparsity/SparsityCompute.hpp
)
set(ZENSIM_LIBRARY_IO_INCLUDE_FILES
    io/IO.h
    io/MeshIO.hpp
    io/ParticleIO.hpp
    io/stb_image_write.h
    io/stb_image.h
    io/Utility.hpp
    # simulation
    simulation/init/Boundary.hpp
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
    cuda/CudaFunction.cuh
    cuda/CudaLaunchConfig.cuh
    cuda/AuxiliaryLibs.cuh
    cuda/Allocators.cuh
    cuda/profile/CudaTimers.cuh
    cuda/memory/MemoryPolicy.cuh
    cuda/memory/operations/Copy.hpp
    cuda/math/bit/SpatialEncoder.cuh
    cuda/math/matrix/EigenDecomposition.cuh
    cuda/math/matrix/Givens.cuh
    cuda/math/matrix/qr.cuh
    cuda/math/matrix/svd.cuh
    cuda/math/matrix/Matrix.hpp # recent
    cuda/execution/ExecutionPolicy.cuh
    # container
    cuda/container/HashTable.hpp
    cuda/container/HashTable.cuh
    cuda/container/LookupTable.cuh
    cuda/container/MatrixHash.cuh
    cuda/container/bvh/Bvh.cuh
    cuda/container/Vector.hpp # recent
    cuda/geometry/Structurefree.hpp # recent
    cuda/algorithm/MappingKernels.cuh
    cuda/algorithm/SortKernels.cuh
    # simulation
    cuda/simulation/transfer/P2G.hpp
    cuda/simulation/transfer/G2P.hpp
    cuda/simulation/grid/GridOp.hpp
    cuda/simulation/sparsity/SparsityCompute.hpp
    # physics
    cuda/physics/ConstitutiveModel.hpp
    #
    cuda/geometry/SparseLevelSet.hpp
)
set(ZENSIM_LIBRARY_OMP_INCLUDE_FILES
    omp/execution/ExecutionPolicy.hpp
    omp/math/matrix/MatrixCtor.hpp
    omp/math/matrix/MatrixTransform.hpp
)

add_library(zensim INTERFACE)
set(ZENSIM_CORE_SOURCE_FILES)

add_cpp_library(zscore)
set(ZENSIM_CORE_SOURCE_FILES ${ZENSIM_CORE_SOURCE_FILES}
    ${ZENSIM_LIBRARY_CORE_SOURCE_FILES}
    ${ZENSIM_LIBRARY_OMP_SOURCE_FILES}
)
target_compile_options(zscore 
  PRIVATE     $<$<COMPILE_LANGUAGE:CXX>: -O3 -fopenmp># -fuse-ld=lld -fvisibility=hidden># -flto=thin -fsanitize=cfi 
)
# -fsanitize=address memory thread
set_target_properties(zscore
  PROPERTIES  LINKER_LANGUAGE CXX
              POSITION_INDEPENDENT_CODE ON
              CUDA_ARCHITECTURES OFF
              LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(zscore
  PUBLIC      zsproj_cxx_deps
)
target_sources(zscore PRIVATE 
    ${ZENSIM_LIBRARY_CORE_SOURCE_FILES}
    ${ZENSIM_LIBRARY_OMP_SOURCE_FILES}
)
if (ENABLE_PCH)
target_precompile_headers(zscore INTERFACE 
    ${ZENSIM_LIBRARY_CORE_INCLUDE_FILES}
    ${ZENSIM_LIBRARY_OMP_INCLUDE_FILES}
)
endif()
target_link_libraries(zensim INTERFACE zscore)  #

# openvdb, partio
add_cpp_library(zstool)
target_sources(zstool PRIVATE ${ZENSIM_LIBRARY_TOOL_SOURCE_FILES} ${ZENSIM_LIBRARY_IO_SOURCE_FILES})
if (ENABLE_PCH)
target_precompile_headers(zstool INTERFACE ${ZENSIM_LIBRARY_TOOL_INCLUDE_FILES} ${ZENSIM_LIBRARY_IO_INCLUDE_FILES})
endif()
target_link_libraries(zstool INTERFACE zspartio zscore)
target_link_libraries(zstool PRIVATE OpenVDB::openvdb)
target_link_libraries(zensim INTERFACE zstool)  #

# related to cuda
if (ZS_ENABLE_CUDA)

# ad-hoc (temporary)
target_compile_definitions(zscore PRIVATE ZS_ENABLE_CUDA=1)
set_source_files_properties(resource/Resource.cpp PROPERTIES  LANGUAGE CUDA)
set_source_files_properties(${ZENSIM_LIBRARY_CUDA_SOURCE_FILES} ${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES} PROPERTIES  LANGUAGE CUDA)
set(ZENSIM_CORE_SOURCE_FILES ${ZENSIM_CORE_SOURCE_FILES} ${ZENSIM_LIBRARY_CUDA_SOURCE_FILES})
target_link_libraries(zscore INTERFACE cuda_runtime cuda cudadevrt nvrtc zscore cusolver cublas cusparse zsproj_cuda_deps)
if (ENABLE_PCH)
target_precompile_headers(zscore INTERFACE ${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES})
endif()
target_compile_options(zscore 
  PRIVATE       $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} -dlto -fopenmp --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread --use_fast_math -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
)
target_link_options(zscore 
  PRIVATE       $<$<LINK_LANGUAGE:CUDA>:-dlto>
)
set_target_properties(zscore
  PROPERTIES  CUDA_EXTENSIONS ON
              CUDA_SEPARABLE_COMPILATION ON
              LINKER_LANGUAGE CUDA
              #CUDA_RESOLVE_DEVICE_SYMBOLS OFF
              LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
              POSITION_INDEPENDENT_CODE ON
              CUDA_ARCHITECTURES OFF
)
target_compile_definitions(zscore 
  PRIVATE     CMAKE_GENERATOR_PLATFORM=x64
)

endif()

target_sources(zscore PRIVATE ${ZENSIM_CORE_SOURCE_FILES})