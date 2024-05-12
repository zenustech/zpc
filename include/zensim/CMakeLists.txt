set(ZENSIM_LIBRARY_CORE_SOURCE_FILES
  math/probability/Probability.cpp
  math/matrix/MatrixTransform.cpp
  memory/MemOps.cpp
  memory/Allocator.cpp
  profile/CppTimers.cpp
  execution/Stacktrace.cpp
  # execution/ExecutionPolicy.cpp
  execution/ConcurrencyPrimitive.cpp
  types/Iterator.cpp
  Logger.cpp
  io/IO.cpp

  #
  visitors/ObjectVisitor.cpp

  #
  ZpcImplPattern.cpp

  io/Filesystem.cpp
  # simulation
)
set(ZENSIM_LIBRARY_IO_SOURCE_FILES
  io/ParticleIO.cpp
)
set(ZENSIM_LIBRARY_TOOL_SOURCE_FILES
  geometry/SparseGrid_Conversion.cpp
  geometry/AdaptiveGrid_Conversion.cpp
  geometry/VdbLevelSet_Point.cpp
  geometry/VdbLevelSet_LoadPhi.cpp
  geometry/VdbLevelSet_LoadPhiVel.cpp
  geometry/VdbLevelSet_Conversion.cpp
  geometry/VdbSampler.cpp

  # simulation
  simulation/init/Scene.cpp
)
set(ZENSIM_LIBRARY_VULKAN_SOURCE_FILES
  vulkan/VkUtils.cpp
  vulkan/VkCommand.cpp
  vulkan/VkTexture.cpp
  vulkan/VkShader.cpp
  vulkan/VkSwapchain.cpp
  vulkan/VkPipeline.cpp
  vulkan/VkContext.cpp
  vulkan/VkContextQuery.cpp
  vulkan/Vulkan.cpp
)
set(ZENSIM_LIBRARY_CUDA_SOURCE_FILES
  cuda/Cuda.cu
  cuda/profile/CudaTimers.cu
  cuda/memory/MemOps.cpp
  cuda/memory/Allocator.cpp
  cuda/simulation/wrangler/Wrangler.cpp
)
set(ZENSIM_LIBRARY_OMP_SOURCE_FILES
  omp/math/matrix/MatrixTransform.cpp
  # omp/execution/ExecutionPolicy.cpp
)
set(ZENSIM_LIBRARY_ZPC_SOURCE_FILES
  # ref: https://stackoverflow.com/questions/2351148/explicit-template-instantiation-when-is-it-used
  container/HashTable.cpp
  container/Bht.cpp
  container/Vector.cpp
  container/TileVector.cpp
  container/DenseField.cpp
  geometry/Mesh.cpp
  math/matrix/Matrix.cpp
  simulation/mpm/Simulator.cpp
  simulation/sparsity/SparsityCompute.cpp
  resource/Resource.cpp
  ZpcInitialization.cpp
)
set(ZENSIM_LIBRARY_ZPC_VULKAN_SOURCE_FILES
  vulkan/VkModel.cpp
)
#set(ZENSIM_LIBRARY_ZPC_CUDA_SOURCE_FILES

  # cuda/math/matrix/Matrix.cpp
#  cuda/simulation/sparsity/SparsityCompute.cpp
#)
set(ZENSIM_LIBRARY_PY_INTEROP_SOURCE_FILES
  py_interop/Allocator.cpp
  py_interop/DenseFieldInstantiations.cpp
  py_interop/BhtInstantiations.cpp
  py_interop/VectorInstantiations.cpp
  py_interop/SpmatInstantiations.cpp
  py_interop/TileVectorInstantiations.cpp
  py_interop/SmallVecInstantiations.cpp
)

# # headers
set(ZENSIM_LIBRARY_CORE_INCLUDE_FILES

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
  geometry/Mesh.hpp
  geometry/PoissonDisk.hpp
  geometry/SparseLevelSet.hpp
  geometry/LevelSetUtils.hpp

  # math
  math/bit/Bits.h
  math/curve/InterpolationKernel.hpp
  math/linear/ConjugateResidual.hpp
  math/linear/ConjugateGradient.hpp
  math/linear/MinimumResidual.hpp
  math/linear/LinearOperators.hpp
  math/matrix/MatrixUtils.h
  math/matrix/Utility.h
  math/matrix/MatrixTransform.hpp
  math/matrix/MatrixCtor.hpp
  math/matrix/Transform.hpp
  math/matrix/Matrix.hpp
  math/matrix/Givens.hpp
  math/matrix/QRSVD.hpp
  math/matrix/SVD.hpp
  math/probability/Probability.h
  math/Hash.hpp
  math/MathUtils.h
  math/RandomNumber.hpp
  math/Rotation.hpp
  math/VecInterface.hpp
  math/Vec.h
  math/Tensor.hpp

  # memory
  memory/MemOps.hpp
  memory/Allocator.h
  memory/MemoryResource.h

  # meta
  ZpcMeta.hpp
  ZpcFunctional.hpp
  meta/Functional.h
  meta/Meta.h
  meta/Sequence.h

  # visitors
  visitors/ObjectVisitor.hpp

  # profile
  profile/CppTimers.hpp

  # types
  types/Pointers.hpp
  types/Iterator.h
  types/Optional.h
  types/Polymorphism.h
  types/Property.h
  types/Tuple.h
  types/BuilderBase.hpp
  types/SmallVector.hpp
  types/SourceLocation.hpp
  types/View.h
  Logger.hpp
  Platform.hpp
  Reflection.h
  # Singleton.h
  TypeAlias.hpp

  # simulation
  physics/constitutive_models/AnisotropicArap.hpp
  physics/constitutive_models/EquationOfState.hpp
  physics/constitutive_models/StvkWithHencky.hpp
  physics/constitutive_models/FixedCorotated.h
  physics/constitutive_models/NeoHookean.hpp
  physics/plasticity_models/SnowPlasticity.hpp
  physics/plasticity_models/VonMisesCapped.hpp
  physics/plasticity_models/NonAssociativeDruckerPrager.hpp
  physics/ConstitutiveModel_Vol_dP.hpp
  physics/ConstitutiveModel.hpp
  physics/SoundSpeedCfl.hpp
  simulation/mpm/Simulator.hpp
  simulation/transfer/P2G.hpp
  simulation/transfer/G2P.hpp
  simulation/transfer/G2P2G.hpp
  simulation/transfer/P2C2G.hpp
  simulation/transfer/G2C2P.hpp
  simulation/grid/GridOp.hpp
  simulation/particle/ParticleOp.hpp
  simulation/sparsity/SparsityOp.hpp
  simulation/sparsity/SparsityCompute.hpp
  simulation/Utils.hpp

  # tpp
  simulation/sparsity/SparsityCompute.tpp

  io/Filesystem.hpp
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
set(ZENSIM_LIBRARY_VULKAN_INCLUDE_FILES
  vulkan/Vulkan.hpp
)
set(ZENSIM_LIBRARY_CUDA_INCLUDE_FILES
  cuda/Cuda.h
  cuda/DeviceUtils.cuh
  cuda/HostUtils.hpp
  cuda/CudaLaunchConfig.cuh
  cuda/profile/CudaTimers.cuh
  cuda/memory/MemOps.hpp
  cuda/memory/Allocator.h
  cuda/math/matrix/EigenDecomposition.cuh
  cuda/math/matrix/svd.cuh
  cuda/execution/ExecutionPolicy.cuh

  # container
  cuda/geometry/Structurefree.hpp # recent
  cuda/algorithm/MappingKernels.cuh
  cuda/algorithm/SortKernels.cuh

  # simulation
  cuda/simulation/transfer/P2G.hpp

  # physics
  cuda/physics/ConstitutiveModel.hpp

  # wrangler
  cuda/simulation/wrangler/Wrangler.hpp
)
set(ZENSIM_LIBRARY_OMP_INCLUDE_FILES
  omp/execution/ExecutionPolicy.hpp
  omp/math/matrix/MatrixTransform.hpp
)
set(ZENSIM_LIBRARY_ZPC_INCLUDE_FILES
  # container
  container/DenseGrid.hpp
  container/RingBuffer.hpp
  container/TileVector.hpp
  container/HashTable.hpp
  container/Vector.hpp
  container/Bvh.hpp
  container/Bvtt.hpp
  container/Bht.hpp
  container/Bcht.hpp
  container/IndexBuckets.hpp
  container/RBTreeMap.hpp
  geometry/PointDataGrid.hpp
  geometry/OverlayGrid.hpp
  geometry/AdaptiveGrid.hpp
  geometry/Structure.hpp
  geometry/Structurefree.hpp
  geometry/SparseGrid.hpp
  math/matrix/SparseMatrix.hpp
  math/matrix/SparseMatrixOperations.hpp
  graph/ConnectedComponents.hpp

  # resource
  resource/Resource.h
  Zpc.hpp

)
set(ZENSIM_LIBRARY_PY_INTEROP_INCLUDE_FILES
  py_interop/DenseFieldView.hpp
  py_interop/VectorView.hpp
  py_interop/TileVectorView.hpp
  py_interop/BhtView.hpp
  py_interop/HashUtils.hpp
  py_interop/SmallVec.hpp
)

if(ZS_ENABLE_CUDA)
  set(ZENSIM_LIBRARY_ZPC_INCLUDE_FILES ${ZENSIM_LIBRARY_ZPC_INCLUDE_FILES}
    cuda/math/matrix/Matrix.hpp
  )
endif()

if(ZS_ENABLE_OMP)
  set(ZENSIM_LIBRARY_ZPC_INCLUDE_FILES ${ZENSIM_LIBRARY_ZPC_INCLUDE_FILES}
  )
endif()

# wrangler
set(ZENSIM_LIBRARY_CUDA_WRANGLER_INCLUDE_FILES
)
set(ZENSIM_LIBRARY_CUDA_WRANGLER_SOURCE_FILES
  # cuda/simulation/wrangler/VolumeWrangler.cpp
  cuda/simulation/wrangler/ParticleWrangler.cpp
)

# set(CMAKE_CXX_VISIBILITY_PRESET     hidden)
# set(CMAKE_VISIBILITY_INLINES_HIDDEN NO)
# include(GenerateExportHeader)
# generate_export_header(zensim EXPORT_FILE_NAME ZensimExport.hpp)
# message("building shared?(${BUILD_SHARED_LIBS}) library")
add_library(zensim INTERFACE)

# ############
# zpc: stable#
# ############
add_library(zpcbase INTERFACE)
target_sources(zpcbase 
  INTERFACE

  ZpcBuiltin.hpp
  ZpcFunction.hpp
  ZpcFunctional.hpp
  ZpcImplPattern.hpp
  ZpcIntrinsics.hpp
  ZpcIterator.hpp
  ZpcMathUtils.hpp
  ZpcMeta.hpp
  ZpcReflection.hpp
  ZpcResource.hpp
  ZpcTuple.hpp

  types/Property.h
  types/SmallVector.hpp

  math/Tensor.hpp
  math/Vec.h

  execution/Atomics.hpp
  # types/Mask.hpp
)
target_include_directories(zpcbase INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(zpcbase INTERFACE zpc_cxx_deps)

# ############
# zpc: core #
# ############
if(ZS_BUILD_SHARED_LIBS)
  add_library(zpccore SHARED)
  target_compile_definitions(zpccore PRIVATE ZPC_BACKEND_PRIVATE)
else()
  add_library(zpccore STATIC)
endif()

target_sources(zpccore
  PRIVATE
  ${ZENSIM_LIBRARY_CORE_SOURCE_FILES} ${ZENSIM_LIBRARY_CORE_INCLUDE_FILES}
)
if(ZS_ENABLE_VULKAN)
  target_sources(zpccore
    PRIVATE
    ${ZENSIM_LIBRARY_VULKAN_SOURCE_FILES} ${ZENSIM_LIBRARY_VULKAN_INCLUDE_FILES}
  )
endif(ZS_ENABLE_VULKAN)

if(ENABLE_PCH)
  target_precompile_headers(zpccore INTERFACE
    ${ZENSIM_LIBRARY_CORE_INCLUDE_FILES}
  )
endif()

target_link_libraries(zpccore PUBLIC zpc_cxx_deps zpcbase)
target_link_libraries(zpccore PUBLIC zswhereami)
# if (WIN32)
#   target_link_libraries(zpccore PUBLIC synchronization)
# endif(WIN32)
target_compile_options(zpccore
  PUBLIC $<$<COMPILE_LANGUAGE:CXX>: $<IF:$<CXX_COMPILER_ID:MSVC>, /Zc:__cplusplus /utf-8 /bigobj $<IF:$<CONFIG:Debug>, , /O2> /EHsc, $<IF:$<CXX_COMPILER_ID:Clang>, -Xclang -O3, -O3> >> # -fuse-ld=lld -fvisibility=hidden># -flto=thin -fsanitize=cfi
)
target_link_options(zpccore
  PUBLIC $<HOST_LINK:
  $<IF:$<CXX_COMPILER_ID:MSVC>, /MACHINE:X64 /OPT:REF /OPT:ICF,
  $<IF:$<CXX_COMPILER_ID:GNU>, "LINKER:--no-as-needed" , >
  $<IF:$<CXX_COMPILER_ID:Clang>, ,>
  -O3 -m64>>
)
set_target_properties(zpccore
  PROPERTIES POSITION_INDEPENDENT_CODE ON

  # CXX_VISIBILITY_PRESET default
)

# target_link_libraries(zpc_cxx_deps INTERFACE $<IF:$<CXX_COMPILER_ID:MSVC>, synchronization.lib, >)

if(ZS_ENABLE_OPENMP)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_OPENMP=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_OPENMP=0)
endif(ZS_ENABLE_OPENMP)

if(ZS_ENABLE_CUDA)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_CUDA=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_CUDA=0)
endif(ZS_ENABLE_CUDA)

if(ZS_ENABLE_VULKAN)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_VULKAN=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_VULKAN=0)
endif(ZS_ENABLE_VULKAN)

if(ZS_ENABLE_SYCL_COMPUTECPP)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL_COMPUTECPP=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL_COMPUTECPP=0)
endif(ZS_ENABLE_SYCL_COMPUTECPP)

if(ZS_ENABLE_SYCL_DPCPP)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL_DPCPP=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL_DPCPP=0)
endif(ZS_ENABLE_SYCL_DPCPP)

if(ZS_ENABLE_SYCL)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL=0)
endif(ZS_ENABLE_SYCL)

# ##############
# zpc: openmp #
# ##############
if(ZS_ENABLE_OPENMP)
  if(ZS_BUILD_SHARED_LIBS)
    add_library(zpcomp SHARED)
    target_compile_definitions(zpcomp PRIVATE ZPC_BACKEND_PRIVATE)
  else()
    add_library(zpcomp STATIC)
  endif()

  target_sources(zpcomp
    PRIVATE ${ZENSIM_LIBRARY_OMP_SOURCE_FILES} ${ZENSIM_LIBRARY_OMP_INCLUDE_FILES}
  )

  if(ENABLE_PCH)
    target_precompile_headers(zpcomp INTERFACE ${ZENSIM_LIBRARY_OMP_INCLUDE_FILES})
  endif()

  set_target_properties(zpcomp
    PROPERTIES POSITION_INDEPENDENT_CODE ON
  )

  target_link_libraries(zpcomp PUBLIC zpc_omp_deps zpccore)

  if(WIN32)
    target_compile_options(zpcomp # MSVC, GNU, Clang, Intel
      # PUBLIC $<$<COMPILE_LANGUAGE:CXX>: /openmp:llvm /openmp:experimental>
      PUBLIC $<$<COMPILE_LANGUAGE:CXX>: /openmp>
    )
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("/openmp:experimental" MSVC_SUPPORT_OPENMP_EXPERIMENTAL)
    if (MSVC_SUPPORT_OPENMP_EXPERIMENTAL)
      target_compile_options(zpcomp
        PUBLIC $<$<COMPILE_LANGUAGE:CXX>: /openmp:experimental>
      )
    endif()
  else()
    target_compile_options(zpcomp # MSVC, GNU, Clang, Intel
      PUBLIC $<$<COMPILE_LANGUAGE:CXX>: $<IF:$<CXX_COMPILER_ID:AppleClang>, -Xpreprocessor -fopenmp, -fopenmp> >
    )
  endif()

  if(WIN32 AND ZS_BUILD_SHARED_LIBS)
    add_custom_command(
      TARGET zpcomp 
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      -DOBJECTS=$<TARGET_RUNTIME_DLLS:zpcomp>
      -DOUTPUT=$<TARGET_FILE_DIR:zpcomp>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cuda/copy_ptx.cmake

      VERBATIM
      DEPENDS zpcomp 
        COMMENT "copying zpc omp deps"
    )
  endif(WIN32 AND ZS_BUILD_SHARED_LIBS)
endif(ZS_ENABLE_OPENMP)

# ############
# zpc: cuda #
# ############
if(ZS_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(ZS_BUILD_SHARED_LIBS)
    add_library(zpccuda SHARED)
    target_compile_definitions(zpccuda PRIVATE ZPC_BACKEND_PRIVATE)
  else()
    add_library(zpccuda STATIC)
  endif()

  target_sources(zpccuda
    PRIVATE ${ZENSIM_LIBRARY_CUDA_SOURCE_FILES} # ${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES}
  )

  if(ENABLE_PCH)
    target_precompile_headers(zpccuda INTERFACE ${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES})
  endif()

  set_source_files_properties(${ZENSIM_LIBRARY_CUDA_SOURCE_FILES} ${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES} PROPERTIES LANGUAGE CUDA)

  set_target_properties(zpccuda
    PROPERTIES POSITION_INDEPENDENT_CODE ON
  )

  target_link_libraries(zpccuda PUBLIC zpc_cuda_deps zpccore CUDA::nvrtc) # cuda_runtime cuda cudadevrt nvrtc cusolver cublas cusparse
  target_link_libraries(zpccuda PRIVATE zpc_cuda_arch_options)

  target_compile_options(zpccuda
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} -lineinfo --ptxas-options=-allow-expensive-optimizations=true>

    # https://stackoverflow.com/questions/64523302/cuda-missing-return-statement-at-end-of-non-void-function-in-constexpr-if-fun
    # https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior
    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda --expt-relaxed-constexpr --default-stream=per-thread $<IF:$<CXX_COMPILER_ID:MSVC>, -Xcompiler="/bigobj", > -Xcudafe "--display_error_number --diag_suppress=implicit_return_from_non_void_function --diag_suppress=loop_not_reachable --diag_suppress=2642 --diag_suppress=1394" > # -dlto -Xptxas -dlcm=cg
    # lto issue: https://forums.developer.nvidia.com/t/using-dlink-time-opt-together-with-gencode-in-cmake/165224
  )

  # target_link_options(zpccuda
  #   PRIVATE $<DEVICE_LINK:-dlto>
  # )

  # -dlto $<IF:$<AND:$<BOOL:${ZS_AUTO_DETECT_CUDA_ARCH}>,$<VERSION_GREATER_EQUAL:${CUDAToolkit_VERSION},11.5.1>>,-arch=native,>>

  # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#implicit-cuda-host-code
  # https://gitlab.kitware.com/cmake/cmake/-/issues/18265
  # target_link_options(zpccuda PRIVATE $<DEVICE_LINK:-dlto>)

  set_target_properties(zpccuda
    PROPERTIES # LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON # https://devblogs.nvidia.com/building-cuda-applications-cmake/

    # CUDA_VISIBILITY_PRESET default
  )
  target_compile_definitions(zpc_cuda_deps INTERFACE CMAKE_GENERATOR_PLATFORM=x64)
endif(ZS_ENABLE_CUDA)

# ############
# zpc: sycl #
# ############
if(ZS_ENABLE_SYCL_COMPUTECPP)
  add_library(zpcsycl INTERFACE)
  target_link_libraries(zpcsycl INTERFACE zpc_sycl_deps zpc_cxx_deps)

elseif(ZS_ENABLE_SYCL_DPCPP)
  add_library(zpcsycl INTERFACE)
  target_link_libraries(zpcsycl INTERFACE zpc_sycl_deps zpc_cxx_deps)

elseif(ZS_ENABLE_SYCL)
  add_library(zpcsycl INTERFACE)
  target_link_libraries(zpcsycl INTERFACE zpc_sycl_deps zpc_cxx_deps)
endif(ZS_ENABLE_SYCL_COMPUTECPP)

# ############
# zpc: cl   #
# ############
if(ZS_ENABLE_OPENCL)
  add_library(zpccl INTERFACE)
  target_link_libraries(zpccl INTERFACE zpc_cl_deps zpc_cxx_deps)
endif(ZS_ENABLE_OPENCL)

# ######
# zpc #
# ######
if(ZS_BUILD_SHARED_LIBS)
  add_library(zpc SHARED)
  target_compile_definitions(zpc PRIVATE ZPC_PRIVATE)
else()
  add_library(zpc STATIC)
endif()

target_sources(zpc
  PRIVATE
  ${ZENSIM_LIBRARY_ZPC_SOURCE_FILES}
  ${ZENSIM_LIBRARY_ZPC_INCLUDE_FILES}
  resource/ExecutionPolicy.cpp
)
set_target_properties(zpc
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(zpc PUBLIC zpccore)

if(ZS_ENABLE_VULKAN)

  target_sources(zpc
    PRIVATE ${ZENSIM_LIBRARY_ZPC_VULKAN_SOURCE_FILES}
  )
endif(ZS_ENABLE_VULKAN)

if(ZS_ENABLE_OPENMP)
  target_link_libraries(zpc PUBLIC zpcomp)

  target_sources(zpc
    PRIVATE resource/omp/ExecutionPolicy.cpp
  )

  # target_sources(zpc PUBLIC $<TARGET_OBJECTS:zpcomp>)
  # target_link_libraries(zpc PUBLIC zpc_omp_deps)
endif(ZS_ENABLE_OPENMP)

if(ZS_ENABLE_CUDA)
  # set_source_files_properties(${ZENSIM_LIBRARY_ZPC_CUDA_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
  target_link_libraries(zpc PUBLIC zpccuda)

  target_sources(zpc
    PRIVATE resource/cuda/ExecutionPolicy.cu
  )
  # target_sources(zpc PRIVATE $<TARGET_OBJECTS:zpccuda>)
  # target_link_libraries(zpc PUBLIC zpc_cuda_deps)
endif(ZS_ENABLE_CUDA)

if(ZS_ENABLE_SYCL_COMPUTECPP)
  target_link_libraries(zpc PUBLIC zpcsycl)
elseif(ZS_ENABLE_SYCL_DPCPP)
  target_link_libraries(zpc PUBLIC zpcsycl)
elseif(ZS_ENABLE_SYCL)
  target_link_libraries(zpc PUBLIC zpcsycl)
endif(ZS_ENABLE_SYCL_COMPUTECPP)

if(ZS_ENABLE_OPENCL)
  target_link_libraries(zpc PUBLIC zpccl)
endif(ZS_ENABLE_OPENCL)

target_link_libraries(zensim INTERFACE zpc) # zpc -> zensim
if (ZS_ENABLE_CUDA)
  if (ZS_AUTO_DETECT_CUDA_ARCH)
    target_link_libraries(zensim INTERFACE zpc_cuda_arch_options)
  endif(ZS_AUTO_DETECT_CUDA_ARCH)
endif(ZS_ENABLE_CUDA)

# ################
# zpc_py_interop #
# ################
add_library(zpc_jit_py INTERFACE)  # cmake helper target for zpc JIT
set(ZS_PYTHON_FOUND 0)
if (NOT ZS_OVERWRITE_PYTHON_INCLUDE_DIR STREQUAL "" AND NOT ZS_OVERWRITE_PYTHON_LIBRARIES STREQUAL "")
  set(ZS_PYTHON_FOUND 1)
  # set(Python3_LIBRARIES ${ZS_OVERWRITE_PYTHON_LIBRARIES})
  # set(Python3_INCLUDE_DIRS ${ZS_OVERWRITE_PYTHON_INCLUDE_DIR})
  if (ZS_OVERWRITE_PYTHON_EXECUTABLE STREQUAL "")
    cmake_path(GET ZS_OVERWRITE_PYTHON_INCLUDE_DIR PARENT_PATH PYTHON_ENV_PATH)
    if (WIN32)
      set(ZS_OVERWRITE_PYTHON_EXECUTABLE ${PYTHON_ENV_PATH}/python.exe)
    else(WIN32)
      cmake_path(GET PYTHON_ENV_PATH PARENT_PATH PYTHON_ENV_PATH)
      set(ZS_OVERWRITE_PYTHON_EXECUTABLE ${PYTHON_ENV_PATH}/bin/python)
    endif(WIN32)
  endif()

else()
  find_package(Python3 COMPONENTS Interpreter Development)
  if (Python3_FOUND)
    set(ZS_PYTHON_FOUND 1)
    set(ZS_OVERWRITE_PYTHON_LIBRARIES ${Python3_LIBRARIES})
    set(ZS_OVERWRITE_PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})
    set(ZS_OVERWRITE_PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
  endif(Python3_FOUND)
endif()

set(ZS_PYTHON_FOUND ${ZS_PYTHON_FOUND} PARENT_SCOPE)

if (ZS_PYTHON_FOUND)
  message(STATUS "python3 libraries for zpc_jit_py: ${ZS_OVERWRITE_PYTHON_LIBRARIES}")
  message(STATUS "python3 include directories for zpc_jit_py: ${ZS_OVERWRITE_PYTHON_INCLUDE_DIR}")
  message(STATUS "python3 executable for zpc_jit_py: ${ZS_OVERWRITE_PYTHON_EXECUTABLE}")
  target_link_libraries(zpc_jit_py INTERFACE ${ZS_OVERWRITE_PYTHON_LIBRARIES})
  target_include_directories(zpc_jit_py INTERFACE ${ZS_OVERWRITE_PYTHON_INCLUDE_DIR})

  set(ZS_OVERWRITE_PYTHON_LIBRARIES ${ZS_OVERWRITE_PYTHON_LIBRARIES} PARENT_SCOPE)
  set(ZS_OVERWRITE_PYTHON_INCLUDE_DIR ${ZS_OVERWRITE_PYTHON_INCLUDE_DIR} PARENT_SCOPE)
  set(ZS_OVERWRITE_PYTHON_EXECUTABLE ${ZS_OVERWRITE_PYTHON_EXECUTABLE} PARENT_SCOPE)
endif(ZS_PYTHON_FOUND)

#[=[
# other overridden setup
## a
# find_package(Python3 3.7 REQUIRED)
# find_package(Python3 3.7 EXACT REQUIRED)

## b
set(Python3_LIBRARIES C:/Develop/miniconda3/envs/py38/libs/python38.lib)
set(Python3_INCLUDE_DIRS C:/Develop/miniconda3/envs/py38/include)

message(STATUS "python3 libraries for zpc_jit_py: ${Python3_LIBRARIES}")
message(STATUS "python3 include directories for zpc_jit_py: ${Python3_INCLUDE_DIRS}")
target_link_libraries(zpc_jit_py INTERFACE ${Python3_LIBRARIES})
target_include_directories(zpc_jit_py INTERFACE ${Python3_INCLUDE_DIRS})
#]=]

add_library(zpc_py_interop SHARED)  # 
set_target_properties(zpc_py_interop 
  PROPERTIES 
  POSITION_INDEPENDENT_CODE ON
  #
  # ref:https://stackoverflow.com/questions/32028667/are-cmake-source-dir-and-project-source-dir-the-same-in-cmake
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  #
  # ref: https://stackoverflow.com/questions/58360502/how-to-set-rpath-origin-in-cmake
  # ref: https://stackoverflow.com/questions/57915564/cmake-how-to-set-rpath-to-origin-with-cmake
  SKIP_BUILD_RPATH OFF
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "\$ORIGIN"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)
target_sources(zpc_py_interop 
  PRIVATE
  ${ZENSIM_LIBRARY_PY_INTEROP_SOURCE_FILES}
  ${ZENSIM_LIBRARY_PY_INTEROP_INCLUDE_FILES}
  py_interop/serial/ExecutionPolicy.cpp
  py_interop/serial/TileVectorUtility.cpp
  py_interop/serial/SpmatUtility.cpp
  py_interop/serial/BhtUtility.cpp
)
if(ZS_ENABLE_OPENMP)
  target_sources(zpc_py_interop
    PRIVATE
    py_interop/omp/ExecutionPolicy.cpp
    py_interop/omp/TileVectorUtility.cpp
    py_interop/omp/SpmatUtility.cpp
    py_interop/omp/BhtUtility.cpp
  )
endif(ZS_ENABLE_OPENMP)
target_link_libraries(zpc_py_interop PUBLIC zpc)
# target_link_libraries(zpc_jit_py INTERFACE zpc_py_interop)

if(ZS_ENABLE_JIT)
add_library(zpc_jit_clang SHARED)
set_target_properties(zpc_jit_clang 
  PROPERTIES 
  POSITION_INDEPENDENT_CODE ON
  #
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  #
  SKIP_BUILD_RPATH OFF
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "\$ORIGIN"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)
target_sources(zpc_jit_clang  
  PRIVATE
  # llvm/LLVM.cpp

  py_interop/Clang.cpp  # may include omp backend
)
target_link_libraries(zpc_jit_clang PRIVATE zpc zpc_jit_deps)
# target_link_libraries(zpc_jit_py INTERFACE zpc_jit_clang)
endif(ZS_ENABLE_JIT)

if(ZS_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(zpc_jit_nvrtc SHARED)
  set_target_properties(zpc_jit_nvrtc 
    PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF  # single file
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    #
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    #
    SKIP_BUILD_RPATH OFF
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )
  target_sources(zpc_jit_nvrtc
    PRIVATE
    py_interop/cuda/Nvrtc.cpp
  )
  target_link_libraries(zpc_jit_nvrtc PRIVATE zpc CUDA::nvrtc)
  # target_link_libraries(zpc_jit_py INTERFACE zpc_jit_nvrtc)

  target_sources(zpc_py_interop
    PRIVATE
    py_interop/cuda/ExecutionPolicy.cpp
    py_interop/cuda/TileVectorUtility.cpp
    py_interop/cuda/SpmatUtility.cpp
    py_interop/cuda/BhtUtility.cpp
  )
  set_source_files_properties(
    py_interop/cuda/ExecutionPolicy.cpp 
    py_interop/cuda/TileVectorUtility.cpp
    py_interop/cuda/SpmatUtility.cpp
    py_interop/cuda/BhtUtility.cpp
    py_interop/cuda/Nvrtc.cpp
    PROPERTIES LANGUAGE CUDA)
endif(ZS_ENABLE_CUDA)

if(ENABLE_PCH)
  target_precompile_headers(zpc_py_interop INTERFACE ${ZENSIM_LIBRARY_PY_INTEROP_INCLUDE_FILES})
endif(ENABLE_PCH)

add_dependencies(zensim zpc_py_interop)
# do not link the target (zpc_py_interop) to zensim!

### general JIT-dependent shared library installation
if(ZS_ENABLE_JIT)
  add_dependencies(
      zpc_jit_py zpc_jit_clang
  )
  if (ZS_PYTHON_FOUND)
    add_custom_command(
      TARGET zpc_py_interop
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      -DOBJECTS=${ZS_OVERWRITE_PYTHON_LIBRARIES}
      -DOUTPUT=$<TARGET_FILE_DIR:zpc_py_interop>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cuda/copy_ptx.cmake
      VERBATIM
      DEPENDS zpc_py_interop 
        COMMENT "copying zpc_jit_py related python runtime shared libraries ${ZS_OVERWRITE_PYTHON_LIBRARIES}"
    )
  endif(ZS_PYTHON_FOUND)

  if (ZS_BUILD_SHARED_LIBS OR WHEREAMI_BUILD_SHARED_LIBS)
    add_custom_command(
      TARGET zpc_jit_clang
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:zpc_jit_clang>
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:zswhereami> $<TARGET_FILE_DIR:zpc_jit_clang>

      VERBATIM
      DEPENDS zpc_jit_clang 
        COMMENT "copying JIT-dependent 3rd party runtime shared libraries"
    )
    find_package(LLVM CONFIG)
    find_package(Clang CONFIG HINTS "${LLVM_DIR}/../clang" "${LLVM_DIR}/../lib/cmake/clang")
    get_target_property(jit_dep_libs zpc_jit_deps INTERFACE_LINK_LIBRARIES)
    foreach(lib ${jit_dep_libs})
      if (TARGET ${lib})
        get_target_property(lib_type ${lib} TYPE)
        if (lib_type STREQUAL "SHARED_LIBRARY")
          # message(STATUS "########\t${lib}: ${lib_type}")
          add_custom_command(
            TARGET zpc_jit_clang
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${lib}> $<TARGET_FILE_DIR:zpc_jit_clang>/$<TARGET_FILE_BASE_NAME:${lib}>$<TARGET_FILE_SUFFIX:${lib}>
            VERBATIM
            DEPENDS zpc_jit_clang 
              COMMENT "copying JIT-dependent llvm/clang runtime shared libraries"
          )
        endif(lib_type STREQUAL "SHARED_LIBRARY")
      endif(TARGET ${lib})
    endforeach(lib)

  endif(ZS_BUILD_SHARED_LIBS OR WHEREAMI_BUILD_SHARED_LIBS)

  if (ZS_BUILD_SHARED_LIBS)
    add_custom_command(
      TARGET zpc_jit_clang
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:zpccore> $<TARGET_FILE_DIR:zpc_jit_clang>
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:zpc> $<TARGET_FILE_DIR:zpc_jit_clang>

      DEPENDS zpc_jit_clang 
        COMMENT "copying JIT-dependent zpc core runtime shared libraries"
  )
  if (ZS_ENABLE_OPENMP)
    add_custom_command(
      TARGET zpc_jit_clang
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:zpcomp> $<TARGET_FILE_DIR:zpc_jit_clang>

      DEPENDS zpc_jit_clang 
        COMMENT "copying JIT-dependent zpc (omp) runtime shared libraries"
    )
  endif(ZS_ENABLE_OPENMP)
  if (ZS_ENABLE_CUDA)
    add_custom_command(
      TARGET zpc_jit_clang
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:zpccuda> $<TARGET_FILE_DIR:zpc_jit_clang>

      DEPENDS zpc_jit_clang 
        COMMENT "copying JIT-dependent zpc (cuda) runtime shared libraries"
    )
  endif(ZS_ENABLE_CUDA)
  endif(ZS_BUILD_SHARED_LIBS)
endif(ZS_ENABLE_JIT)

### nvrtc-specific shared library installation
if (ZS_ENABLE_CUDA)
  add_dependencies(
      zpc_jit_py zpc_jit_nvrtc 
  )
  get_target_property(nvrtc_target_type CUDA::nvrtc TYPE)
  if (nvrtc_target_type STREQUAL "UNKNOWN_LIBRARY" OR nvrtc_target_type STREQUAL "SHARED_LIBRARY")
  add_custom_command(
    TARGET zpc_jit_nvrtc
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:zpc_jit_nvrtc>
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:CUDA::nvrtc> $<TARGET_FILE_DIR:zpc_jit_nvrtc>
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:CUDA::cuda_driver> $<TARGET_FILE_DIR:zpc_jit_nvrtc>
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:CUDA::cudart> $<TARGET_FILE_DIR:zpc_jit_nvrtc>

    VERBATIM
    DEPENDS zpc_jit_nvrtc 
      COMMENT "copying nvrtc related runtime shared libraries"
  )
#[=[
if (TARGET CUDA::nvJitLink)
  add_custom_command(
    TARGET zpc_jit_nvrtc
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:CUDA::nvJitLink> $<TARGET_FILE_DIR:zpc_jit_nvrtc>

    VERBATIM
    DEPENDS zpc_jit_nvrtc 
    COMMENT "copying nvJitLink shared library"
  )
  endif()
#]=]
  endif (nvrtc_target_type STREQUAL "UNKNOWN_LIBRARY" OR nvrtc_target_type STREQUAL "SHARED_LIBRARY")
endif(ZS_ENABLE_CUDA)

# openvdb, partio
# ##########
# zpctool  #
# ##########
if(ZS_BUILD_SHARED_LIBS)
  add_library(zpctool SHARED)
  target_compile_definitions(zpctool PRIVATE ZPC_EXTENSION_PRIVATE)
else()
  add_library(zpctool STATIC)
endif()

target_sources(zpctool
  PRIVATE
  ${ZENSIM_LIBRARY_IO_SOURCE_FILES}
  ${ZENSIM_LIBRARY_IO_INCLUDE_FILES}
)

if(ENABLE_PCH)
  target_precompile_headers(zpctool INTERFACE ${ZENSIM_LIBRARY_IO_INCLUDE_FILES})
endif(ENABLE_PCH)

target_link_libraries(zpctool PUBLIC zpc)

set_target_properties(zpctool
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)

if(ZS_ENABLE_PARTIO)
  target_link_libraries(zpctool PUBLIC zspartio)
endif(ZS_ENABLE_PARTIO)

if(ZS_ENABLE_MSHIO)
  target_link_libraries(zpctool PUBLIC mshio::mshio)
endif(ZS_ENABLE_MSHIO)

# set_property(TARGET zpctool APPEND PROPERTY PUBLIC_HEADER "${ZENSIM_LIBRARY_TOOL_INCLUDE_FILES} ${ZENSIM_LIBRARY_IO_INCLUDE_FILES}")
if(ZS_ENABLE_OPENVDB)
  if(ENABLE_PCH)
    target_precompile_headers(zpctool INTERFACE ${ZENSIM_LIBRARY_TOOL_INCLUDE_FILES})
  endif(ENABLE_PCH)

  target_link_libraries(zpctool PRIVATE zsopenvdb)
  target_sources(zpctool
    PRIVATE
    ${ZENSIM_LIBRARY_TOOL_SOURCE_FILES}
    ${ZENSIM_LIBRARY_TOOL_INCLUDE_FILES}
  )
endif(ZS_ENABLE_OPENVDB)

if (WIN32 AND ZS_BUILD_SHARED_LIBS)
  add_custom_command(
    TARGET zpctool 
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    -DOBJECTS=$<TARGET_RUNTIME_DLLS:zpctool>
    -DOUTPUT=$<TARGET_FILE_DIR:zpctool>
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cuda/copy_ptx.cmake

    VERBATIM
    DEPENDS zpctool  
      COMMENT "copying zpc tool deps"
  )
endif(WIN32 AND ZS_BUILD_SHARED_LIBS)

target_link_libraries(zensim INTERFACE zpctool) # zpctool -> zensim

# #################
# zs_cu_wrangler #
# #################
if(ZS_ENABLE_CUDA AND ZS_ENABLE_ZENO_CU_WRANGLE)
  add_library(zs_cu_wrangler OBJECT)

  target_sources(zs_cu_wrangler
    PRIVATE ${ZENSIM_LIBRARY_CUDA_WRANGLER_SOURCE_FILES}
  )
  target_compile_options(zs_cu_wrangler
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-ptx -lineinfo --ptxas-options=-allow-expensive-optimizations=true> # -ptx --keep
  )
  set_source_files_properties(${ZENSIM_LIBRARY_CUDA_WRANGLER_INCLUDE_FILES} ${ZENSIM_LIBRARY_CUDA_WRANGLER_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)

  target_link_libraries(zs_cu_wrangler PRIVATE zpc CUDA::nvrtc)
  set(ZS_PTX_SOURCE_DIR cuda/simulation/wrangler)

  # set(ZS_PTX_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/zs_cu_wrangler.dir/${ZS_PTX_SOURCE_DIR})  # not valid for win platform
  # set(ZS_PTX_INCLUDE_DIR ${PROJECT_BINARY_DIR}/${ZS_PTX_SOURCE_DIR})
  set(ZS_PTX_INCLUDE_DIR ${RESOURCE_BASE_DIR}/resource)
  # message(FATAL_ERROR "to set ZS_PTX_INCLUDE_DIR: [${CMAKE_RUNTIME_OUTPUT_DIRECTORY}]")
  set_target_properties(zs_cu_wrangler
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON # https://devblogs.nvidia.com/building-cuda-applications-cmake/
    CUDA_PTX_COMPILATION ON # https://stackoverflow.com/questions/55993745/how-should-i-get-cmake-to-also-create-ptx-files-for-my-kernels
    CUDA_ARCHITECTURES 52

    # CUDA_VISIBILITY_PRESET default
  )
#[=[
  get_target_property(cuwrangler_compile_flags zs_cu_wrangler COMPILE_OPTIONS)
  separate_arguments(cuwrangler_compile_flags)
  foreach(flag ${cuwrangler_compile_flags})
    message(STATUS "checking zs cu wrangler flag [${flag}]")
    if(${flag} MATCHES "^-gencode=arch=compute_*")
      message(STATUS " -> removing flag [${flag}] for zs_cu_wrangler!")
      list(REMOVE_ITEM cuwrangler_compile_flags ${flag})
    endif()
  endforeach()
  set_property(TARGET zs_cu_wrangler PROPERTY COMPILE_OPTIONS ${cuwrangler_compile_flags})

# ref: developer.nvidia.com/blog/building-cuda-applications-cmake/
find_program(bin_to_c
  NAMES bin2c
  PATHS ${CUDAToolkit_BIN_DIR}
)
if(NOT bin_to_c)
  message(FATAL_ERROR
    "bin2c not found:\n"
    " CUDAToolkit_BIN_DIR='${CUDAToolkit_BIN_DIR}'\n"
  )
endif()

set(zs_cu_wrangler_output_file ${CMAKE_CURRENT_SOURCE_DIR}/${ZS_PTX_SOURCE_DIR}/wrangler_ptxs.h)

add_custom_command(
  OUTPUT  "${zs_cu_wrangler_output_file}"
  COMMAND ${CMAKE_COMMAND}
    -DBIN_TO_C_COMMAND=${bin_to_c}
    -DOBJECTS=$<TARGET_OBJECTS:zs_cu_wrangler>
    -DOUTPUT=${zs_cu_wrangler_output_file}
    -P bin2c_wrapper.cmake
  VERBATIM
  DEPENDS zpctool
  COMMENT "converting ptx files to a header"
)
#]=]
  add_custom_command(
    TARGET zpctool
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ZS_PTX_INCLUDE_DIR}
    COMMAND ${CMAKE_COMMAND}
    -DOBJECTS=$<TARGET_OBJECTS:zs_cu_wrangler>
    -DOUTPUT=${ZS_PTX_INCLUDE_DIR}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cuda/copy_ptx.cmake

    # COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_OBJECTS:zs_cu_wrangler> ${ZS_PTX_INCLUDE_DIR}
    VERBATIM
    DEPENDS zpctool
    COMMENT "copying wrangler ptx files"
  )

  target_compile_definitions(zpc_cuda_deps INTERFACE ZS_PTX_INCLUDE_DIR="${ZS_PTX_INCLUDE_DIR}")

  # $<TARGET_OBJECTS:zs_cu_wrangler>
  add_dependencies(zpctool zs_cu_wrangler)
  target_link_libraries(zensim INTERFACE zs_cu_wrangler) # zs_cu_wrangler -> zensim
endif(ZS_ENABLE_CUDA AND ZS_ENABLE_ZENO_CU_WRANGLE)

# poisson disk resource file
add_custom_command(
  TARGET zpctool
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${RESOURCE_BASE_DIR}/resource
  COMMAND ${CMAKE_COMMAND}
  -DOBJECTS=${CMAKE_CURRENT_SOURCE_DIR}/../../zpc_assets/MpmParticles/particles-1000k.dat
  -DOUTPUT=${RESOURCE_BASE_DIR}/resource
  -P ${CMAKE_CURRENT_SOURCE_DIR}/cuda/copy_ptx.cmake

  VERBATIM
  DEPENDS zpc
    COMMENT "copying resource file [particles-1000k.dat] for poisson disk sampling"
)

# resource file
add_custom_command(
  TARGET zpctool
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${RESOURCE_BASE_DIR}/resource

  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/../../zpc_assets/Fonts
  ${RESOURCE_BASE_DIR}/resource/fonts

  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/../../zpc_assets/Icons
  ${RESOURCE_BASE_DIR}/resource/icons

  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/../../zpc_assets/TriMesh
  ${RESOURCE_BASE_DIR}/resource/models

  VERBATIM
  DEPENDS zpc
    COMMENT "copying resource files (fonts, icons, textures...) for gui"
)

# python helloworld script
add_custom_command(
  TARGET zpctool
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${RESOURCE_BASE_DIR}/resource
  COMMAND ${CMAKE_COMMAND}
  -DOBJECTS=${CMAKE_CURRENT_SOURCE_DIR}/../../zpc_assets/Scripts/HelloWorld.py
  -DOUTPUT=${RESOURCE_BASE_DIR}/resource
  -P ${CMAKE_CURRENT_SOURCE_DIR}/cuda/copy_ptx.cmake

  VERBATIM
  DEPENDS zpc
    COMMENT "copying script file [HelloWorld.py] for py module"
)

# for header installation
# list_transform_prepend(ZENSIM_INTERFACE_FILES include/zensim/)
# message("${ZENSIM_INTERFACE_FILES}")
# PUBLIC_HEADER not working as expected.
# set_target_properties(zensim PROPERTIES PUBLIC_HEADER "${ZENSIM_INTERFACE_FILES}")
