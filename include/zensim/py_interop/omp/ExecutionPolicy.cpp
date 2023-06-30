#include "zensim/omp/execution/ExecutionPolicy.hpp"
#define Zensim_EXPORT
#include "zensim/ZensimExport.hpp"

extern "C" {

ZENSIM_EXPORT zs::OmpExecutionPolicy *policy__parallel() { return new zs::OmpExecutionPolicy; }
ZENSIM_EXPORT void del_policy__parallel(zs::OmpExecutionPolicy *v) { delete v; }
}