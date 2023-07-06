#include "zensim/omp/execution/ExecutionPolicy.hpp"

extern "C" {

zs::OmpExecutionPolicy *policy__parallel() { return new zs::OmpExecutionPolicy; }
void del_policy__parallel(zs::OmpExecutionPolicy *v) { delete v; }
}