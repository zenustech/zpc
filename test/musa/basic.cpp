#if 0
#  include "zensim/musa/Musa.h"

int main() {
  using namespace zs;
  (void)Musa::instance();
  return 0;
}

#else
#  include <musa.h>
#  include <musa_runtime.h>

#  include <iostream>

#  include "zensim/musa/Musa.h"

template <size_t... Is> void tttt(std::index_sequence<Is...>) {
  ((void)(std::cout << Is << ','), ...);
}

template <typename T> constexpr T add(T a, T b) { return a + b; }

template <typename Fn> __global__ void test(Fn fn) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  printf("[%d]: %f, %f\n", idx, add((float)idx, ::sqrt(idx * 1.2f)), fn(idx));
}

int main() {
  if constexpr (true) {
    tttt(std::make_index_sequence<5>{});
    std::cout << "succeed" << std::endl;
  } else {
    std::cout << "failed" << std::endl;
  }
  using namespace zs;
  (void)Musa::instance();
  MUresult result = muInit(0);
  if (result != MUSA_SUCCESS) {
    std::cout << "[muInit] MUSA error: " << result << '\n';
    return -1;
  }

  int numTotalDevice = 0;
  result = muDeviceGetCount(&numTotalDevice);
  if (result != MUSA_SUCCESS) {
    std::cout << "[muDeviceGetCount] MUSA error: " << result << '\n';
    return -1;
  } else if (numTotalDevice == 0) {
    std::cout << "No musa device detected" << '\n';
    return -1;
  }

  for (int i = 0; i < numTotalDevice; ++i) {
    MUdevice dev = 0;
    MUcontext ctx;
    muDeviceGet(&dev, i);

    int major, minor;
    result = muDeviceGetAttribute(&minor, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    if (result != MUSA_SUCCESS) {
      std::cout << "[muDeviceGetAttribute: minor] MUSA error: " << result << '\n';
      return -1;
    }
    result = muDeviceGetAttribute(&major, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    if (result != MUSA_SUCCESS) {
      std::cout << "[muDeviceGetAttribute: major] MUSA error: " << result << '\n';
      return -1;
    }
    unsigned int const compute_capability = major * 10 + minor;
    std::cout << compute_capability << ' ';
  }

  test<<<1, 32>>>([a = 1] __device__(int x) -> float { return x * x; });
  musaDeviceSynchronize();

  return 0;
}

#endif
