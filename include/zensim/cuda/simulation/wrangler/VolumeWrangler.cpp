#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structure.hpp"
#include "zensim/types/Tuple.h"

#if 0
extern __device__ float user_script(int n);

extern "C" __global__ void zpc_prebuilt_kernel() {
  auto id = zs::make_tuple(blockIdx.x, threadIdx.x);
  printf("what the heck, indeed called successfully at (%d, %d), user specified result: %f!\n",
         (int)id.template get<0>(), (int)id.template get<1>(), user_script(threadIdx.x));
}
#endif
