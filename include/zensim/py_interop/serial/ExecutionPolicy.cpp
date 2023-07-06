#include "zensim/execution/ExecutionPolicy.hpp"

extern "C" {

zs::SequentialExecutionPolicy *policy__serial() {
  return new zs::SequentialExecutionPolicy;
}
void del_policy__serial(zs::SequentialExecutionPolicy *v) { delete v; }
}