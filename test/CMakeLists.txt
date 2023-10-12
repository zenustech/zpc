# tuple
add_executable(tupletest tuple.cpp)
target_link_libraries(tupletest PRIVATE zpc)

add_test(Tuple tupletest)
add_dependencies(zensim tupletest)

# parallel primitives
add_executable(paraprim parallel_primitives.cpp)
target_link_libraries(paraprim PRIVATE zpc)

add_test(ParallelPrimitive paraprim)
add_dependencies(zensim paraprim)