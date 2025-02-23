# tuple
add_executable(tupletest tuple.cpp)
target_link_libraries(tupletest PRIVATE zpc)

add_test(ZsTuple tupletest)
add_dependencies(zensim tupletest)

# map
add_executable(maptest map.cpp)
target_link_libraries(maptest PRIVATE zpc)

add_test(ZsRBTreeMap maptest)
add_dependencies(zensim maptest)

# helperlexer
add_executable(helperlexer help_parser.cpp)
target_link_libraries(helperlexer PRIVATE zpc)

add_test(HelperLexer helperlexer)
add_dependencies(zensim helperlexer)

# nd vec
add_executable(ndarraytest ndvec.cpp)
target_link_libraries(ndarraytest PRIVATE zpc)

add_test(ZsNdArray ndarraytest)
add_dependencies(zensim ndarraytest)

# parallel primitives
add_executable(paraprim parallel_primitives.cpp)
target_link_libraries(paraprim PRIVATE zpc)

add_test(ZsParallelPrimitive paraprim)
add_dependencies(zensim paraprim)

# binary search
add_executable(binarysearchtest binary_search.cpp)
target_link_libraries(binarysearchtest PRIVATE zpc)

add_test(ZsBinarySearch binarysearchtest)
add_dependencies(zensim binarysearchtest)

# sycl backend
if(ZS_ENABLE_SYCL_ONEAPI OR ZS_ENABLE_SYCL_ACPP)
    #
    add_executable(sycltests sycl/main.cpp)
    target_link_libraries(sycltests PRIVATE zpc zpc_sycl_deps)

    add_test(ZsSyclTests sycltests)
    add_dependencies(zensim sycltests)
endif()
#

# cuda backend
if(ZS_ENABLE_CUDA)
    #
    add_executable(cutests cuda/main.cu)
    target_link_libraries(cutests PRIVATE zpc)

    add_test(ZsCudaTests cutests)
    add_dependencies(zensim cutests)
    #
    add_executable(cucontainer cuda/basic.cu)
    target_link_libraries(cucontainer PRIVATE zpc)

    add_test(ZsCudaBasic cucontainer)
    add_dependencies(zensim cucontainer)
endif(ZS_ENABLE_CUDA)

# musa backend
if(ZS_ENABLE_MUSA)
    #
    zs_add_musa_executable(mutests SOURCES musa/basic.cpp DEVICE_SOURCES musa/basic.cpp)
    # add_executable(mutests musa/basic.cpp)
    target_link_libraries(mutests PRIVATE zpc)

    add_test(ZsMusaBasic mutests)
    add_dependencies(zensim mutests)
endif(ZS_ENABLE_MUSA)
