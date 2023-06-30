#include "zensim/execution/ExecutionPolicy.hpp"
#define Zensim_EXPORT
#include "zensim/ZensimExport.hpp"

extern "C" {

ZENSIM_EXPORT zs::SequentialExecutionPolicy *policy__serial() {
  return new zs::SequentialExecutionPolicy;
}
ZENSIM_EXPORT void del_policy__serial(zs::SequentialExecutionPolicy *v) { delete v; }
}