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

# cuda backends
if (ZS_ENABLE_CUDA)
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
endif (ZS_ENABLE_CUDA)