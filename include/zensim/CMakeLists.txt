set(ZENSIM_LIBRARY_CORE_SOURCE_FILES
    DynamicLoader.cpp
    types/Config.cpp
    types/Node.cpp
    math/probability/Probability.cpp
    math/matrix/MatrixTransform.cpp
    math/matrix/MatrixCtor.cpp
    math/matrix/Matrix.cpp
    memory/Allocator.cpp
    profile/CppTimers.cpp
    execution/Stacktrace.cpp
    execution/ExecutionNode.cpp
    execution/ExecutionPolicy.cpp
    resource/Resource.cpp
    # simulation
    simulation/mpm/Simulator.cpp
)
set(ZENSIM_LIBRARY_IO_SOURCE_FILES
    io/IO.cpp
    io/stb_image.cpp
    # simulation
    simulation/init/Boundary.cpp
    simulation/init/Scene.cpp
)
set(ZENSIM_LIBRARY_TOOL_SOURCE_FILES
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
    # execution
    execution/Concurrency.h
    execution/ExecutionNode.h
    execution/ExecutionPolicy.hpp
    execution/Executor.h
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
    types/Config.h
    types/Event.hpp
    types/Function.h
    types/Inherit.h
    types/Iterator.h
    types/Node.h
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
    DynamicLoader.h
    Logger.hpp
    Platform.hpp
    Reflection.h
    Singleton.h
    TypeAlias.hpp
    zensim.hpp
    # simulation
    simulation/mpm/Simulator.hpp
    simulation/init/Structure.hpp
    simulation/init/Structurefree.hpp
    simulation/physics/ConstitutiveModel.hpp
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
    cuda/math/bit/SpatialEncoder.cuh
    cuda/math/matrix/EigenDecomposition.cuh
    cuda/math/matrix/Givens.cuh
    cuda/math/matrix/qr.cuh
    cuda/math/matrix/svd.cuh
    cuda/execution/ExecutionPolicy.cuh
    cuda/container/HashTable.hpp
    cuda/container/HashTable.cuh
    cuda/container/LookupTable.cuh
    cuda/container/MatrixHash.cuh
    cuda/container/bvh/Bvh.cuh
    cuda/algorithm/MappingKernels.cuh
    cuda/algorithm/SortKernels.cuh

    cuda/math/matrix/Matrix.hpp
)
set(ZENSIM_LIBRARY_OMP_INCLUDE_FILES
    omp/execution/ExecutionPolicy.hpp
    omp/math/matrix/MatrixCtor.hpp
    omp/math/matrix/MatrixTransform.hpp
)

add_cpp_library(zscore)
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

# related to openvdb
add_cpp_library(zsvdb)
target_sources(zsvdb PRIVATE ${ZENSIM_LIBRARY_TOOL_SOURCE_FILES})
if (ENABLE_PCH)
target_precompile_headers(zsvdb INTERFACE ${ZENSIM_LIBRARY_TOOL_INCLUDE_FILES})
endif()
target_link_libraries(zsvdb PRIVATE OpenVDB::openvdb)

# related to partio
add_cpp_library(zsio)
target_sources(zsio PRIVATE ${ZENSIM_LIBRARY_IO_SOURCE_FILES})
target_link_libraries(zsio INTERFACE zspartio zscore zsvdb)
if (ENABLE_PCH)
target_precompile_headers(zsio INTERFACE ${ZENSIM_LIBRARY_IO_INCLUDE_FILES})
endif()

# related to cuda
if (ZS_ENABLE_CUDA)

set_source_files_properties(${ZENSIM_LIBRARY_CUDA_SOURCE_FILES} PROPERTIES  LANGUAGE CUDA)
add_cuda_library(zscuda)
target_sources(zscuda PRIVATE ${ZENSIM_LIBRARY_CUDA_SOURCE_FILES})
target_link_libraries(zscuda INTERFACE cuda_runtime cuda cudadevrt nvrtc zscore cusolver cublas cusparse)
if (ENABLE_PCH)
target_precompile_headers(zscuda INTERFACE ${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES})
endif()

endif()
