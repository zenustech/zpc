add_executable(tupletest)
target_sources(tupletest
    PRIVATE     tuple.cpp
)
target_link_libraries(tupletest PRIVATE zensim)

add_test(Tuple tupletest)