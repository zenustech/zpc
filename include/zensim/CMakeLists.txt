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
set(ZENSIM_LIBRARY_SYCL_SOURCE_FILES
  sycl/Sycl.cpp
)
set(ZENSIM_LIBRARY_MUSA_SOURCE_FILES
  musa/Musa.mu
  musa/profile/MusaTimers.mu
  musa/memory/MemOps.mu
  musa/memory/Allocator.mu
)
set(ZENSIM_LIBRARY_ZPC_SOURCE_FILES
  # ref: https://stackoverflow.com/questions/2351148/explicit-template-instantiation-when-is-it-used
  container/HashTable.cpp
  container/Bht.cpp
  container/Vector.cpp
  container/TileVector.cpp
  container/DenseField.cpp
  geometry/SparseGrid.cpp
  geometry/AdaptiveGrid.cpp
  geometry/Collider.cpp

  geometry/Mesh.cpp
  io/MeshIO.cpp

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
  # math/linear/ConjugateResidual.hpp
  # math/linear/ConjugateGradient.hpp
  # math/linear/MinimumResidual.hpp
  # math/linear/LinearOperators.hpp
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
  meta/Functional.h
  meta/Meta.h

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
  # types/View.h
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
  # simulation/sparsity/SparsityCompute.tpp

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
set(ZENSIM_LIBRARY_SYCL_INCLUDE_FILES
  sycl/Sycl.hpp
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
  geometry/Collider.h
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

if(ZS_ENABLE_PCH)
  target_precompile_headers(zpcbase INTERFACE
    # ZpcBuiltin.hpp
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
  )
endif(ZS_ENABLE_PCH)
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
  target_compile_definitions(zpccore PRIVATE ZPC_CORE_PRIVATE)
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

if(ZS_ENABLE_PCH)
  target_precompile_headers(zpccore INTERFACE
    ${ZENSIM_LIBRARY_CORE_INCLUDE_FILES}
  )
endif(ZS_ENABLE_PCH)

target_link_libraries(zpccore PUBLIC zpcbase)
target_link_libraries(zpccore PRIVATE zswhereami)
# if (WIN32)
#   target_link_libraries(zpccore PUBLIC synchronization)
# endif(WIN32)
target_compile_options(zpccore
  PUBLIC $<$<COMPILE_LANGUAGE:CXX>: $<IF:$<CXX_COMPILER_ID:MSVC,Intel>, /Zc:__cplusplus /utf-8 /bigobj $<IF:$<CONFIG:Debug>, , /O2> /EHsc, $<IF:$<CXX_COMPILER_ID:Clang>, -Xclang -O3, -O3> >> # -fuse-ld=lld -fvisibility=hidden># -flto=thin -fsanitize=cfi
)
target_link_options(zpccore
  PUBLIC $<HOST_LINK:
  $<IF:$<CXX_COMPILER_ID:MSVC,Intel>, /MACHINE:X64 /OPT:REF /OPT:ICF,
  $<IF:$<CXX_COMPILER_ID:GNU>, "LINKER:--no-as-needed" , >
  $<IF:$<CXX_COMPILER_ID:Clang>, ,>
  -O3 -m64>>
)
set_target_properties(zpccore
  PROPERTIES POSITION_INDEPENDENT_CODE ON
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

if(ZS_ENABLE_MUSA)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_MUSA=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_MUSA=0)
endif(ZS_ENABLE_MUSA)

if(ZS_ENABLE_JIT)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_JIT=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_JIT=0)
endif(ZS_ENABLE_JIT)

if(ZS_ENABLE_VULKAN)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_VULKAN=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_VULKAN=0)
endif(ZS_ENABLE_VULKAN)

if(ZS_ENABLE_SYCL_ONEAPI)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL_ONEAPI=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL_ONEAPI=0)
endif(ZS_ENABLE_SYCL_ONEAPI)

if(ZS_ENABLE_SYCL_ACPP)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL_ACPP=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL_ACPP=0)
endif(ZS_ENABLE_SYCL_ACPP)

if(ZS_ENABLE_SYCL_ONEAPI OR ZS_ENABLE_SYCL_ACPP)
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL=1)
else()
  target_compile_definitions(zpc_cxx_deps INTERFACE ZS_ENABLE_SYCL=0)
endif()

# ############
# vulkan #
# ############
set(ZS_ENABLE_VULKAN ${ZS_ENABLE_VULKAN} PARENT_SCOPE)

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

  if(ZS_ENABLE_PCH)
    target_precompile_headers(zpcomp INTERFACE ${ZENSIM_LIBRARY_OMP_INCLUDE_FILES})
  endif(ZS_ENABLE_PCH)

  set_target_properties(zpcomp
    PROPERTIES POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON # otherwise fail at runtime(?)
  )

  target_link_libraries(zpcomp PUBLIC zpc_omp_deps zpccore)

  if(WIN32)
    target_compile_options(zpcomp # MSVC, GNU, Clang, Intel
      # PUBLIC $<$<COMPILE_LANGUAGE:CXX>: /openmp:llvm /openmp:experimental>
      PUBLIC $<$<COMPILE_LANGUAGE:CXX>: /openmp>
    )
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("/openmp:experimental" MSVC_SUPPORT_OPENMP_EXPERIMENTAL)
    if(MSVC_SUPPORT_OPENMP_EXPERIMENTAL)
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
      # DEPENDS zpcomp 
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

  # precompiling cuda header files not viable (it seems)
  if(ZS_ENABLE_PCH)
    target_precompile_headers(zpccuda INTERFACE cuda/memory/MemOps.hpp)
    # ${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES}
  endif(ZS_ENABLE_PCH)

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
    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda --expt-relaxed-constexpr --default-stream=per-thread $<IF:$<CXX_COMPILER_ID:MSVC>, -Xcompiler="/bigobj", > -Xcudafe "--display_error_number --diag_suppress=implicit_return_from_non_void_function --diag_suppress=loop_not_reachable --diag_suppress=69 --diag_suppress=2642 --diag_suppress=1394" > # -dlto -Xptxas -dlcm=cg
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
# zpc: musa #
# ############
if(ZS_ENABLE_MUSA)
  find_package(MUSAToolkit REQUIRED)
  find_package(MUSA REQUIRED)

  set(MUSA_COMPILE_OPTIONS -std=c++17 -fPIC -Werror)
  set(MUSA_LINK_OPTIONS)

  get_property(MUSA_ARCH_COMPILE_OPTIONS GLOBAL PROPERTY ZS_MUSA_ARCH_COMPILE_OPTIONS)
  list(APPEND MUSA_COMPILE_OPTIONS ${MUSA_ARCH_COMPILE_OPTIONS})

  #
  # grab include dirs (MUSA_INCLUDE_DIRECTORIES)
  #
  get_target_property(zpcbase_interface_inc_dirs zpcbase INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(zsproj_deps_interface_inc_dirs zsproj_deps INTERFACE_INCLUDE_DIRECTORIES)
  set(MUSA_INCLUDE_DIRECTORIES ${MUSAToolkit_INCLUDE_DIRS} ${zpcbase_interface_inc_dirs} ${zpc_cxx_deps_interface_inc_dirs} ${zsproj_deps_interface_inc_dirs})

  message(STATUS "MUSA include directories: ${MUSA_INCLUDE_DIRECTORIES}")

  #[=[
  #
  # grab link dirs (MUSA_LINK_DIRECTORIES)
  #
  get_target_property(zsproj_deps_interface_link_dirs zsproj_deps INTERFACE_LINK_DIRECTORIES)
  set(MUSA_LINK_DIRECTORIES ${MUSAToolkit_LIBRARY_DIR})
  if(zsproj_deps_interface_link_dirs)
    list(APPEND MUSA_LINK_DIRECTORIES ${zsproj_deps_interface_link_dirs})
  endif(zsproj_deps_interface_link_dirs)

  if(MUSA_LINK_DIRECTORIES)
    foreach(link_dir IN LISTS MUSA_LINK_DIRECTORIES)
      list(APPEND MUSA_LINK_OPTIONS -L${link_dir})
    endforeach()
  endif(MUSA_LINK_DIRECTORIES)

  message(STATUS "MUSA link directories: ${MUSA_LINK_DIRECTORIES}")

  #
  # grab link libs (MUSA_LINK_LIBRARIES)
  #
  get_target_property(zsproj_deps_interface_link_libs zsproj_deps INTERFACE_LINK_LIBRARIES)
  list(APPEND MUSA_LINK_OPTIONS -lmusart)
  if(zsproj_deps_interface_link_libs)
    foreach(link_lib IN LISTS zsproj_deps_interface_link_libs)
      cmake_path(IS_ABSOLUTE link_lib is_absolute_path)
      if(TARGET link_lib)
      elseif(is_absolute_path)
        cmake_path(GET link_lib PARENT_PATH lib_dir)
        cmake_path(GET link_lib FILENAME lib_name)
        list(APPEND MUSA_LINK_OPTIONS -L${lib_dir})
        list(APPEND MUSA_LINK_OPTIONS -l${lib_name})
      else()
        list(APPEND MUSA_LINK_OPTIONS -l${link_lib})
      endif()
    endforeach()
  endif(zsproj_deps_interface_link_libs)
  #]=]

  #
  # grab compile definitions (MUSA_COMPILE_DEFINITIONS)
  #
  #get_target_property(zpcbase_interface_compile_defs zpcbase INTERFACE_COMPILE_DEFINITIONS)
  get_target_property(zpc_cxx_deps_interface_compile_defs zpc_cxx_deps INTERFACE_COMPILE_DEFINITIONS)
  get_target_property(zsproj_deps_interface_compile_defs zsproj_deps INTERFACE_COMPILE_DEFINITIONS)
  set(MUSA_COMPILE_DEFINITIONS ${zpc_cxx_deps_interface_compile_defs} ${zsproj_deps_interface_compile_defs})
  ## not supported so far in MUSA ##
  # list(APPEND MUSA_COMPILE_DEFINITIONS "MUSA_API_PER_THREAD_DEFAULT_STREAM=1")
  string(GENEX_STRIP "${MUSA_COMPILE_DEFINITIONS}" MUSA_COMPILE_DEFINITIONS)
  if(MUSA_COMPILE_DEFINITIONS)
    foreach(def IN LISTS MUSA_COMPILE_DEFINITIONS)
      list(APPEND MUSA_COMPILE_OPTIONS -D${def})
    endforeach()
  endif(MUSA_COMPILE_DEFINITIONS)

  message(STATUS "MUSA compile definitions: ${MUSA_COMPILE_DEFINITIONS}")

  list(JOIN MUSA_COMPILE_OPTIONS " " MUSA_COMPILE_OPTIONS)
  message(STATUS "MUSA compile options: ${MUSA_COMPILE_OPTIONS}")

  # store compile options
  set_property(GLOBAL APPEND PROPERTY ZS_MUSA_COMPILE_OPTIONS ${MUSA_COMPILE_OPTIONS})

  # 
  set(_TRANSFORMED_MUSA_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_transformed_musa_srcs/MUSAComputeCapabilityWorkdir)
  file(MAKE_DIRECTORY ${_TRANSFORMED_MUSA_SOURCE_DIR})
  set_property(GLOBAL APPEND PROPERTY ZS_TRANSFORMED_MUSA_SOURCE_DIR ${_TRANSFORMED_MUSA_SOURCE_DIR})
  #
  # zs_add_musa_library 
  #
  function(zs_add_musa_library targetName libType)
    set(_OPTION_KEYWORDS)
    set(_SINGLE_VAL_KEYWORDS)
    set(_MULTI_VAL_KEYWORDS PRIVATE_DEFINITIONS SOURCES)
    cmake_parse_arguments(ARG
      "${_OPTION_KEYWORDS}" # no value
      "${_SINGLE_VAL_KEYWORDS}" # single value
      "${_MULTI_VAL_KEYWORDS}" # multi value 
      ${ARGN})

    # compile options (definitions, link directory, flags)
    get_property(flags GLOBAL PROPERTY ZS_MUSA_COMPILE_OPTIONS)
    if(${ARG_PRIVATE_DEFINITIONS})
      list(JOIN ARG_PRIVATE_DEFINITIONS " " ARG_PRIVATE_DEFINITIONS)
      set(MUSA_MCC_FLAGS "${flags} ${ARG_PRIVATE_DEFINITIONS} ${ARG_UNPARSED_ARGUMENTS}")
    else()
      set(MUSA_MCC_FLAGS "${flags} ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    # get_property(inc_dirs GLOBAL PROPERTY MUSA_INCLUDE_DIRECTORIES)
    musa_include_directories(${MUSA_INCLUDE_DIRECTORIES})

    # device
    set_source_files_properties(${ARG_SOURCES} MUSA_SOURCE_PROPERTY_FORMAT LANGUAGE OBJ)
    musa_compile(MUSA_OBJS ${ARG_SOURCES})
    # host
    add_library(${targetName} ${libType} ${MUSA_OBJS})
    target_link_libraries(${targetName} PUBLIC zpc_musa_deps zpccore)
    # target_link_options(${targetName} PUBLIC $<HOST_LINK: ${MUSA_LINK_OPTIONS}>)
    set_target_properties(${targetName} PROPERTIES LINKER_LANGUAGE CXX)
    # musa_add_library(${targetName} ${libType} ${ARGN})
  endfunction(zs_add_musa_library)


  if(ZS_BUILD_SHARED_LIBS)
    zs_add_musa_library(zpcmusa SHARED
      SOURCES ${ZENSIM_LIBRARY_MUSA_SOURCE_FILES}
      PRIVATE_DEFINITIONS -DZPC_BACKEND_PRIVATE
    )
  else()
    zs_add_musa_library(zpcmusa STATIC
      SOURCES ${ZENSIM_LIBRARY_MUSA_SOURCE_FILES})
  endif()

  # precompiling cuda header files not viable (it seems)
  if(ZS_ENABLE_PCH)
    # target_precompile_headers(zpcmusa INTERFACE musa/memory/MemOps.hpp)
    # ${ZENSIM_LIBRARY_CUDA_INCLUDE_FILES}
  endif(ZS_ENABLE_PCH)

  # target_link_libraries(zpcmusa PRIVATE zpc_musa_arch_options zpc_musa_deps zpccore MUSA::musart)

  #
  # zs_add_musa_executable
  #
  function(zs_add_musa_executable targetName)
    set(_OPTION_KEYWORDS)
    set(_SINGLE_VAL_KEYWORDS)
    set(_MULTI_VAL_KEYWORDS PRIVATE_DEFINITIONS SOURCES DEVICE_SOURCES)
    cmake_parse_arguments(ARG
      "${_OPTION_KEYWORDS}" # no value
      "${_SINGLE_VAL_KEYWORDS}" # single value
      "${_MULTI_VAL_KEYWORDS}" # multi value 
      ${ARGN})

    # compile options (definitions, link directory, flags)
    get_property(flags GLOBAL PROPERTY ZS_MUSA_COMPILE_OPTIONS)
    if(${ARG_PRIVATE_DEFINITIONS})
      list(JOIN ARG_PRIVATE_DEFINITIONS " " ARG_PRIVATE_DEFINITIONS)
      set(MUSA_MCC_FLAGS "${flags} ${ARG_PRIVATE_DEFINITIONS} ${ARG_UNPARSED_ARGUMENTS}")
    else()
      set(MUSA_MCC_FLAGS "${flags} ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    # get_property(inc_dirs GLOBAL PROPERTY MUSA_INCLUDE_DIRECTORIES)
    musa_include_directories(${MUSA_INCLUDE_DIRECTORIES})

    # NOT WORKING AS EXPECTED!
    #[=[
    if(${ARG_DEVICE_SOURCES})
      set_source_files_properties(${ARG_DEVICE_SOURCES} MUSA_SOURCE_PROPERTY_FORMAT LANGUAGE OBJ)
    endif()
    #]=]
    # ALTERNATIVE APPROACH:
    set(srcs)
    get_property(TRANSFORMED_MUSA_SOURCE_DIR GLOBAL PROPERTY ZS_TRANSFORMED_MUSA_SOURCE_DIR)
    add_custom_target(${targetName}_device_sources)
    foreach(file IN LISTS ARG_SOURCES)
      set(_added FALSE)
      list(FIND ARG_DEVICE_SOURCES ${file} _index)
      if(${_index} GREATER -1)
        # message(FATAL_ERROR "${file} in ${ARG_DEVICE_SOURCES}")
        cmake_path(GET file EXTENSION fileExt)
        if(NOT (fileExt STREQUAL ".mu"))
          set(_prevPath ${file})
          cmake_path(GET file STEM fileStem)
          cmake_path(REMOVE_FILENAME file)
          # cmake_path(APPEND file "${fileStem}.mu")

          file(REMOVE "${TRANSFORMED_MUSA_SOURCE_DIR}/${fileStem}.mu")
          configure_file(${_prevPath} ${TRANSFORMED_MUSA_SOURCE_DIR}/${fileStem}.mu COPYONLY)

          # message(FATAL_ERROR "MUSA: ${_prevPath} -> ${TRANSFORMED_MUSA_SOURCE_DIR}/${fileStem}.mu")
          list(APPEND srcs "${TRANSFORMED_MUSA_SOURCE_DIR}/${fileStem}.mu")
          set(_added TRUE)
          #[=[
          add_custom_command(TARGET ${targetName}_device_sources PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_prevPath}
            ${TRANSFORMED_MUSA_SOURCE_DIR}/${fileStem}.mu
            COMMENT "Copying '${_prevPath}' library to '${TRANSFORMED_MUSA_SOURCE_DIR}/${fileStem}.mu'")
          # message(FATAL_ERROR "MUSA: ${_prevPath} -> ${TRANSFORMED_MUSA_SOURCE_DIR}/${fileStem}.mu")
          #]=]
        endif()
        # else()
        # message(FATAL_ERROR "${file} not in ${ARG_DEVICE_SOURCES}")
      endif()

      if(NOT _added)
        list(APPEND srcs ${file})
      endif()
    endforeach(file IN LISTS ARG_SOURCES)

    # host
    musa_add_executable(${targetName} ${srcs})
    # add_dependencies(${targetName} ${targetName}_device_sources)
    target_link_libraries(${targetName} PUBLIC zpcmusa)
    # target_link_options(${targetName} PUBLIC $<HOST_LINK: ${MUSA_LINK_OPTIONS}>)
    set_target_properties(${targetName} PROPERTIES LINKER_LANGUAGE CXX)
    # musa_add_executable(${targetName} ${ARGN})
  endfunction(zs_add_musa_executable)
endif(ZS_ENABLE_MUSA)

# ###########
# zpc: sycl #
# ###########
if(ZS_ENABLE_SYCL_ONEAPI)
  if(ZS_BUILD_SHARED_LIBS)
    add_library(zpcsycl SHARED)
    target_compile_definitions(zpcsycl PRIVATE ZPC_BACKEND_PRIVATE)

    # https://www.scivision.dev/cmake-shared-libs-visual-studio-windows/
    target_link_options(zpc_cxx_deps INTERFACE
      $<HOST_LINK:
      $<IF:$<CXX_COMPILER_ID:Intel>, --export-all-symbols, >
      >
    )
  else()
    add_library(zpcsycl STATIC)
  endif()

  target_sources(zpcsycl
    PRIVATE ${ZENSIM_LIBRARY_SYCL_SOURCE_FILES}
  )
  target_link_libraries(zpcsycl PUBLIC zpc_sycl_deps zpccore)

elseif(ZS_ENABLE_SYCL_ACPP)
  if(ZS_BUILD_SHARED_LIBS)
    add_library(zpcsycl SHARED)
    target_compile_definitions(zpcsycl PRIVATE ZPC_BACKEND_PRIVATE)
  else()
    add_library(zpcsycl STATIC)
  endif()

  target_sources(zpcsycl
    PRIVATE ${ZENSIM_LIBRARY_SYCL_SOURCE_FILES}
  )
  target_link_libraries(zpcsycl PUBLIC zpc_sycl_deps zpccore)
endif(ZS_ENABLE_SYCL_ONEAPI)

# ############
# zpc: cl    #
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

if(ZS_ENABLE_MUSA)
  # set_source_files_properties(${ZENSIM_LIBRARY_ZPC_CUDA_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
  target_link_libraries(zpc PUBLIC zpcmusa)

  target_sources(zpc
    PRIVATE resource/musa/ExecutionPolicy.mu
  )
endif(ZS_ENABLE_MUSA)

if(ZS_ENABLE_SYCL_ONEAPI)
  target_link_libraries(zpc PUBLIC zpcsycl)
elseif(ZS_ENABLE_SYCL_ACPP)
  target_link_libraries(zpc PUBLIC zpcsycl)
endif(ZS_ENABLE_SYCL_ONEAPI)

if(ZS_ENABLE_OPENCL)
  target_link_libraries(zpc PUBLIC zpccl)
endif(ZS_ENABLE_OPENCL)

target_link_libraries(zensim INTERFACE zpc) # zpc -> zensim
if(ZS_ENABLE_CUDA)
  if(ZS_AUTO_DETECT_CUDA_ARCH)
    target_link_libraries(zensim INTERFACE zpc_cuda_arch_options)
  endif(ZS_AUTO_DETECT_CUDA_ARCH)
endif(ZS_ENABLE_CUDA)

# ################
# zpc_py_interop #
# ################
add_library(zpc_jit_py INTERFACE) # cmake helper target for zpc JIT
set(ZS_PYTHON_FOUND OFF)

if(ZS_ENABLE_PYTHON)

  message(STATUS "\n-- ------------------------------- Checking Python ------------------------------- ")
  if(NOT ZS_OVERWRITE_PYTHON_INCLUDE_DIR STREQUAL "" AND NOT ZS_OVERWRITE_PYTHON_LIBRARIES STREQUAL "" AND EXISTS "${ZS_OVERWRITE_PYTHON_INCLUDE_DIR}/Python.h")
    set(ZS_PYTHON_FOUND ON)
    # set(Python3_LIBRARIES ${ZS_OVERWRITE_PYTHON_LIBRARIES})
    # set(Python3_INCLUDE_DIRS ${ZS_OVERWRITE_PYTHON_INCLUDE_DIR})
    if(ZS_OVERWRITE_PYTHON_EXECUTABLE STREQUAL "")
      cmake_path(GET ZS_OVERWRITE_PYTHON_INCLUDE_DIR PARENT_PATH PYTHON_ENV_PATH)
      if(WIN32)
        set(ZS_OVERWRITE_PYTHON_EXECUTABLE ${PYTHON_ENV_PATH}/python.exe)
      else(WIN32)
        cmake_path(GET PYTHON_ENV_PATH PARENT_PATH PYTHON_ENV_PATH)
        set(ZS_OVERWRITE_PYTHON_EXECUTABLE ${PYTHON_ENV_PATH}/bin/python)
      endif(WIN32)
    endif()

    # ananconda python, vanilla python
    if(WIN32)
      cmake_path(GET ZS_OVERWRITE_PYTHON_INCLUDE_DIR PARENT_PATH ZS_OVERWRITE_PYTHON_RUNTIME_LIBRARY_DIRS)
      set(ZS_OVERWRITE_PYTHON_LINK_DIRECTORY ${ZS_OVERWRITE_PYTHON_RUNTIME_LIBRARY_DIRS}/libs)
    else()
      cmake_path(GET ZS_OVERWRITE_PYTHON_INCLUDE_DIR PARENT_PATH ZS_OVERWRITE_PYTHON_RUNTIME_LIBRARY_DIRS)
      cmake_path(GET ZS_OVERWRITE_PYTHON_RUNTIME_LIBRARY_DIRS PARENT_PATH ZS_OVERWRITE_PYTHON_RUNTIME_LIBRARY_DIRS)
      set(ZS_OVERWRITE_PYTHON_LINK_DIRECTORY ${ZS_OVERWRITE_PYTHON_RUNTIME_LIBRARY_DIRS}/lib)
      set(ZS_OVERWRITE_PYTHON_RUNTIME_LIBRARY_DIRS ${ZS_OVERWRITE_PYTHON_LINK_DIRECTORY})
    endif()

  else()
    find_package(Python3 COMPONENTS Interpreter Development)
    if(Python3_FOUND)
      set(ZS_PYTHON_FOUND ON)
      set(ZS_OVERWRITE_PYTHON_LIBRARIES ${Python3_LIBRARIES})
      set(ZS_OVERWRITE_PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})
      set(ZS_OVERWRITE_PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
      set(ZS_OVERWRITE_PYTHON_LINK_DIRECTORY ${Python3_LIBRARY_DIRS})
      set(ZS_OVERWRITE_PYTHON_RUNTIME_LIBRARY_DIRS ${Python3_RUNTIME_LIBRARY_DIRS})
    endif(Python3_FOUND)
  endif()

  set(ZS_PYTHON_FOUND ${ZS_PYTHON_FOUND} PARENT_SCOPE)

  if(ZS_PYTHON_FOUND)
    message(STATUS "python3 link directory for zpc_jit_py: ${ZS_OVERWRITE_PYTHON_LINK_DIRECTORY}")
    message(STATUS "python3 libraries for zpc_jit_py: ${ZS_OVERWRITE_PYTHON_LIBRARIES}")
    message(STATUS "python3 include directories for zpc_jit_py: ${ZS_OVERWRITE_PYTHON_INCLUDE_DIR}")
    message(STATUS "python3 executable for zpc_jit_py: ${ZS_OVERWRITE_PYTHON_EXECUTABLE}")
    message(STATUS "python3 runtime library directories for zpc_jit_py: ${ZS_OVERWRITE_PYTHON_RUNTIME_LIBRARY_DIRS}")

    set(ZS_PYTHON_DLLS "")
    foreach(lib ${ZS_OVERWRITE_PYTHON_LIBRARIES})
      cmake_path(GET lib EXTENSION LAST_ONLY lib_type)
      if(WIN32)
        if(lib_type STREQUAL ".lib")
          cmake_path(GET lib STEM dll_name)
          if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            if(dll_name MATCHES "_d$")
              foreach(rtdir ${ZS_OVERWRITE_PYTHON_RUNTIME_LIBRARY_DIRS})
                if(EXISTS "${rtdir}/${dll_name}.dll")
                  list(APPEND ZS_PYTHON_DLLS "${rtdir}/${dll_name}.dll")
                endif() # check existence
              endforeach() # iterate runtime library directories
            endif(dll_name MATCHES "_d$") # dll filename has _d suffix
          else(CMAKE_BUILD_TYPE STREQUAL "Debug")
            if(NOT dll_name MATCHES "_d$")
              foreach(rtdir ${ZS_OVERWRITE_PYTHON_RUNTIME_LIBRARY_DIRS})
                if(EXISTS "${rtdir}/${dll_name}.dll")
                  list(APPEND ZS_PYTHON_DLLS "${rtdir}/${dll_name}.dll")
                endif() # check existence
              endforeach() # iterate runtime library directories
            endif(NOT dll_name MATCHES "_d$") # dll filename has no _d suffix
          endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        endif(lib_type STREQUAL ".lib")
      else(WIN32)
        # assume only .a .so .dylib
        set(VALID_LIB_SUFFIXES ".a" ".so" ".dylib")
        set(SHARED_LIB_SUFFIXES ".so" ".dylib")
        if(NOT ${lib_type} IN_LIST VALID_LIB_SUFFIXES)
          message(FATAL_ERROR "Unknown lib suffix [${lib_type}]!")
        endif(NOT ${lib_type} IN_LIST VALID_LIB_SUFFIXES)
        if(${lib_type} IN_LIST SHARED_LIB_SUFFIXES)
          list(APPEND ZS_PYTHON_DLLS ${lib})
        endif(${lib_type} IN_LIST SHARED_LIB_SUFFIXES)
      endif(WIN32)
    endforeach()
    message(STATUS "python3 runtime shared libs for zpc_jit_py: ${ZS_PYTHON_DLLS}")

    target_link_libraries(zpc_jit_py INTERFACE ${ZS_OVERWRITE_PYTHON_LIBRARIES})
    target_include_directories(zpc_jit_py INTERFACE ${ZS_OVERWRITE_PYTHON_INCLUDE_DIR})
    target_link_directories(zpc_jit_py INTERFACE ${ZS_OVERWRITE_PYTHON_LINK_DIRECTORY})

    set(ZS_OVERWRITE_PYTHON_LIBRARIES ${ZS_OVERWRITE_PYTHON_LIBRARIES} PARENT_SCOPE)
    set(ZS_OVERWRITE_PYTHON_INCLUDE_DIR ${ZS_OVERWRITE_PYTHON_INCLUDE_DIR} PARENT_SCOPE)
    set(ZS_OVERWRITE_PYTHON_EXECUTABLE ${ZS_OVERWRITE_PYTHON_EXECUTABLE} PARENT_SCOPE)
    set(ZS_OVERWRITE_PYTHON_LINK_DIRECTORY ${ZS_OVERWRITE_PYTHON_LINK_DIRECTORY} PARENT_SCOPE)
    # useful for windows
    # set(ZS_OVERWRITE_PYTHON_RUNTIME_LIBRARY_DIRS ${ZS_OVERWRITE_PYTHON_RUNTIME_LIBRARY_DIRS} PARENT_SCOPE)
    set(ZS_PYTHON_DLLS ${ZS_PYTHON_DLLS} PARENT_SCOPE)

  else(ZS_PYTHON_FOUND)
    set(ZS_ENABLE_PYTHON OFF)
  endif(ZS_PYTHON_FOUND)

  message(STATUS "############################### ZPC: Python (${ZS_PYTHON_FOUND}) ###############################")

endif(ZS_ENABLE_PYTHON)

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
]=]

add_library(zpc_py_interop SHARED) # 
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

    py_interop/Clang.cpp # may include omp backend
  )
  target_compile_definitions(zpc_jit_clang PUBLIC -DZS_LLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR})
  target_link_libraries(zpc_jit_clang PRIVATE zpc zpc_jit_deps)
  # target_link_libraries(zpc_jit_py INTERFACE zpc_jit_clang)
endif(ZS_ENABLE_JIT)

if(ZS_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(zpc_jit_nvrtc SHARED)
  set_target_properties(zpc_jit_nvrtc
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF # single file
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

if(ZS_ENABLE_PCH)
  target_precompile_headers(zpc_py_interop INTERFACE ${ZENSIM_LIBRARY_PY_INTEROP_INCLUDE_FILES})
endif(ZS_ENABLE_PCH)

add_dependencies(zensim zpc_py_interop)
# do not link the target (zpc_py_interop) to zensim!

### general JIT-dependent shared library installation upon build
if(ZS_ENABLE_JIT)
  add_dependencies(
    zpc_jit_py zpc_jit_clang
  )
  if(ZS_PYTHON_FOUND)
    foreach(lib ${ZS_PYTHON_DLLS})
      add_custom_command(
        TARGET zpc_py_interop
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${lib} $<TARGET_FILE_DIR:zpc_py_interop>
        VERBATIM
        # DEPENDS zpc_py_interop 
        COMMENT "copying zpc_jit_py related python runtime shared library [${lib}]"
      )
    endforeach()
  endif(ZS_PYTHON_FOUND)

  if(ZS_BUILD_SHARED_LIBS OR WHEREAMI_BUILD_SHARED_LIBS)
    add_custom_command(
      TARGET zpc_jit_clang
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:zpc_jit_clang>
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:zswhereami> $<TARGET_FILE_DIR:zpc_jit_clang>

      VERBATIM
      # DEPENDS zpc_jit_clang 
      COMMENT "copying JIT-dependent runtime shared library [$<TARGET_FILE:zswhereami>]"
    )
    find_package(LLVM CONFIG)
    find_package(Clang CONFIG HINTS "${LLVM_DIR}/../clang" "${LLVM_DIR}/../lib/cmake/clang")
    get_target_property(jit_dep_libs zpc_jit_deps INTERFACE_LINK_LIBRARIES)
    foreach(lib ${jit_dep_libs})
      if(TARGET ${lib})
        get_target_property(lib_type ${lib} TYPE)
        if(lib_type STREQUAL "SHARED_LIBRARY")
          # message(STATUS "########\t${lib}: ${lib_type}")
          add_custom_command(
            TARGET zpc_jit_clang
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${lib}> $<TARGET_FILE_DIR:zpc_jit_clang>/$<TARGET_FILE_BASE_NAME:${lib}>$<TARGET_FILE_SUFFIX:${lib}>
            VERBATIM
            # DEPENDS zpc_jit_clang 
            COMMENT "copying JIT-dependent llvm/clang runtime shared libraries [$<TARGET_FILE:${lib}>]"
          )
        endif(lib_type STREQUAL "SHARED_LIBRARY")
      endif(TARGET ${lib})
    endforeach(lib)

  endif(ZS_BUILD_SHARED_LIBS OR WHEREAMI_BUILD_SHARED_LIBS)

  if(ZS_BUILD_SHARED_LIBS)
    add_custom_command(
      TARGET zpc_jit_clang
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:zpccore> $<TARGET_FILE_DIR:zpc_jit_clang>
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:zpc> $<TARGET_FILE_DIR:zpc_jit_clang>

      # DEPENDS zpc_jit_clang 
      COMMENT "copying JIT-dependent zpc core runtime shared libraries"
    )
    if(ZS_ENABLE_OPENMP)
      add_custom_command(
        TARGET zpc_jit_clang
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:zpcomp> $<TARGET_FILE_DIR:zpc_jit_clang>

        # DEPENDS zpc_jit_clang 
        COMMENT "copying JIT-dependent zpc (omp) runtime shared libraries"
      )
    endif(ZS_ENABLE_OPENMP)
    if(ZS_ENABLE_CUDA)
      add_custom_command(
        TARGET zpc_jit_clang
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:zpccuda> $<TARGET_FILE_DIR:zpc_jit_clang>

        # DEPENDS zpc_jit_clang 
        COMMENT "copying JIT-dependent zpc (cuda) runtime shared libraries"
      )
    endif(ZS_ENABLE_CUDA)
  endif(ZS_BUILD_SHARED_LIBS)
endif(ZS_ENABLE_JIT)

### nvrtc-specific shared library installation
if(ZS_ENABLE_CUDA)
  add_dependencies(
    zpc_jit_py zpc_jit_nvrtc
  )
  get_target_property(nvrtc_target_type CUDA::nvrtc TYPE)
  if(nvrtc_target_type STREQUAL "UNKNOWN_LIBRARY" OR nvrtc_target_type STREQUAL "SHARED_LIBRARY")
    add_custom_command(
      TARGET zpc_jit_nvrtc
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:zpc_jit_nvrtc>
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:CUDA::nvrtc> $<TARGET_FILE_DIR:zpc_jit_nvrtc>
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:CUDA::cuda_driver> $<TARGET_FILE_DIR:zpc_jit_nvrtc>
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:CUDA::cudart> $<TARGET_FILE_DIR:zpc_jit_nvrtc>

      VERBATIM
      # DEPENDS zpc_jit_nvrtc 
      COMMENT "copying nvrtc related runtime shared libraries"
    )
    #[=[
if (TARGET CUDA::nvJitLink)
  add_custom_command(
    TARGET zpc_jit_nvrtc
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:CUDA::nvJitLink> $<TARGET_FILE_DIR:zpc_jit_nvrtc>

    VERBATIM
    # DEPENDS zpc_jit_nvrtc 
    COMMENT "copying nvJitLink shared library"
  )
  endif()
]=]
  endif(nvrtc_target_type STREQUAL "UNKNOWN_LIBRARY" OR nvrtc_target_type STREQUAL "SHARED_LIBRARY")
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

set_target_properties(zpctool
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON

  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}

  SKIP_BUILD_RPATH OFF
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "\$ORIGIN"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

target_sources(zpctool
  PRIVATE
  ${ZENSIM_LIBRARY_IO_SOURCE_FILES}
  ${ZENSIM_LIBRARY_IO_INCLUDE_FILES}
)

if(ZS_ENABLE_PCH)
  target_precompile_headers(zpctool INTERFACE ${ZENSIM_LIBRARY_IO_INCLUDE_FILES})
endif(ZS_ENABLE_PCH)

target_link_libraries(zpctool PUBLIC zpc)

set_target_properties(zpctool
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)

if(ZS_ENABLE_PARTIO)
  target_link_libraries(zpctool PRIVATE zspartio)
endif(ZS_ENABLE_PARTIO)

if(ZS_ENABLE_MSHIO)
  target_link_libraries(zpctool PRIVATE mshio::mshio)
endif(ZS_ENABLE_MSHIO)

# set_property(TARGET zpctool APPEND PROPERTY PUBLIC_HEADER "${ZENSIM_LIBRARY_TOOL_INCLUDE_FILES} ${ZENSIM_LIBRARY_IO_INCLUDE_FILES}")
if(ZS_ENABLE_OPENVDB)
  if(ZS_ENABLE_PCH)
    target_precompile_headers(zpctool INTERFACE ${ZENSIM_LIBRARY_TOOL_INCLUDE_FILES})
  endif(ZS_ENABLE_PCH)

  target_link_libraries(zpctool PRIVATE zsopenvdb)
  target_sources(zpctool
    PRIVATE
    ${ZENSIM_LIBRARY_TOOL_SOURCE_FILES}
    ${ZENSIM_LIBRARY_TOOL_INCLUDE_FILES}
  )
endif(ZS_ENABLE_OPENVDB)

if(WIN32 AND ZS_BUILD_SHARED_LIBS)
  add_custom_command(
    TARGET zpctool
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    -DOBJECTS=$<TARGET_RUNTIME_DLLS:zpctool>
    -DOUTPUT=$<TARGET_FILE_DIR:zpctool>
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cuda/copy_ptx.cmake

    VERBATIM
    # DEPENDS zpctool  
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
  # DEPENDS zpctool
  COMMENT "converting ptx files to a header"
)
]=]
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
    # DEPENDS zpctool
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
  # DEPENDS zpc
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
  ${CMAKE_CURRENT_SOURCE_DIR}/../../zpc_assets/Textures
  ${RESOURCE_BASE_DIR}/resource/textures

  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/../../zpc_assets/TriMesh
  ${RESOURCE_BASE_DIR}/resource/models

  VERBATIM
  # DEPENDS zpc
  COMMENT "copying resource files (fonts, icons, textures...) for gui"
)

# python helloworld script
add_custom_command(
  TARGET zpctool
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${RESOURCE_BASE_DIR}/resource
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../../zpc_assets/Scripts/ ${RESOURCE_BASE_DIR}/resource

  VERBATIM
  # DEPENDS zpc
  COMMENT "copying script files [${CMAKE_CURRENT_SOURCE_DIR}/../../zpc_assets/Scripts/]"
)

# for header installation
# list_transform_prepend(ZENSIM_INTERFACE_FILES include/zensim/)
# message("${ZENSIM_INTERFACE_FILES}")
# PUBLIC_HEADER not working as expected.
# set_target_properties(zensim PROPERTIES PUBLIC_HEADER "${ZENSIM_INTERFACE_FILES}")
